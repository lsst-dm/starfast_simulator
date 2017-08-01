# LSST Data Management System
# Copyright 2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""
StarFast is a simple but fast simulation tool to generate images for testing algorithms.
It is optimized for creating many realizations of the same field of sky under different observing conditions.

Four steps to set up the simulation:
1) Create a StarSim object
    example_sim = StarSim(options=options)
2) Load a PSF with example_sim.load_psf(psf, options=options)
    Example psf:
        import galsim
        gsp = galsim.GSParams(folding_threshold=1.0/x_size, maximum_fft_size=12288)
        psf = galsim.Kolmogorov(fwhm=1.0, flux=1, gsparams=gsp)
3) Create a simulated catalog of stars
    example_sim.load_catalog(options=options)
4) Build the raw sky model from the catalog
    example_sim.simulate()

The sky model can be used for many simulations of the same field under different conditions
For each simulated observation convolve the raw sky model with the PSF, and include instrumental,
atmospheric, etc... effects. If desired, a different psf may be supplied for each simulation.
    observed_image = example_sim.convolve(psf=psf, options=options)
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
import os
from scipy import constants
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
# import lsst.afw.coord as afwCoord
from lsst.afw.coord import Coord, IcrsCoord, Observatory
from lsst.afw.geom import Angle
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
from lsst.daf.base import DateTime
import lsst.meas.algorithms as measAlg
import lsst.pex.policy as pexPolicy
from lsst.sims.photUtils import Bandpass, matchStar, PhotometricParameters
from lsst.utils import getPackageDir
from calc_refractive_index import diff_refraction
from fast_dft import fast_dft
import time
import unittest
import lsst.utils.tests

nanFloat = float("nan")
nanAngle = Angle(nanFloat)
lsst_lat = Angle(np.radians(-30.244639))
lsst_lon = Angle(np.radians(-70.749417))
lsst_alt = 2663.


class StarSim:
    """Class that defines a random simulated region of sky, and allows fast transformations."""

    def __init__(self, psf=None, pixel_scale=None, pad_image=1.5, catalog=None, sed_list=None,
                 x_size=512, y_size=512, band_name='g', photons_per_jansky=None,
                 ra=None, dec=None, ra_reference=lsst_lon, dec_reference=lsst_lat,
                 sky_rotation=0.0, exposure_time=30.0, saturation_value=65000,
                 background_level=314, **kwargs):
        """Set up the fixed parameters of the simulation."""
        """
        @param psf: psf object from Galsim. Needs to have methods getFWHM() and drawImage().
        @param pixel_scale: arcsec/pixel to use for final images.
        @param pad_image: Image size padding factor, to reduce FFT aliasing. Set to 1.0 to tile images.
        @param catalog: Supply a catalog from a previous StarSim run. Untested!
        @param sed_list: Supply a list of SEDs from a call to lsst.sims.photUtils matchStar.
                         If None, a simple blackbody radiation spectrum will be used for each source.
        @param x_size: Number of pixels on the x-axis
        @param y_size: Number of pixels on the y-axis
        @param band_name: Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param photons_per_jansky: Conversion factor between Jansky units and photons
        @param ra: Right Ascension of the center of the simulation. Used only for the wcs of output fits files
        @param dec: Declination of the center of the simulation. Used only for the wcs of output fits files.
        @param ra_reference: Right Ascension of the center of the field for the reference catalog
        @param dec_reference: Declination of the center of the field for the reference catalog
        @param sky_rotation: Rotation of the wcs, in degrees.
        @param exposure_time: Length of the exposure, in seconds
        @param saturation_value: Maximum electron counts of the detector before saturation. Turn off with None
        @param background_level: Number of counts to add to every pixel as a sky background.
        """
        bandpass = _load_bandpass(band_name=band_name, **kwargs)
        self.n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.bandpass = bandpass
        self.bandpass_highres = _load_bandpass(band_name=band_name, highres=True, **kwargs)
        self.photParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=pixel_scale,
                                                bandpass=band_name)
        self.band_name = band_name
        if sed_list is None:
            # Load in model SEDs
            matchStarObj = matchStar()
            sed_list = matchStarObj.loadKuruczSEDs()
        self.sed_list = sed_list
        self.catalog = catalog
        self.coord = _CoordsXY(pixel_scale=self.photParams.platescale, pad_image=pad_image,
                               x_size=x_size, y_size=y_size)
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.ExtentI(x_size, y_size))
        self.sky_rotation = sky_rotation
        if ra is None:
            ra = ra_reference
        if dec is None:
            dec = dec_reference
        self.ra = ra
        self.dec = dec
        self.wcs = _create_wcs(pixel_scale=self.coord.scale(), bbox=self.bbox,
                               ra=ra, dec=dec, sky_rotation=sky_rotation)
        self._wcs_ref = _create_wcs(pixel_scale=self.coord.scale(), bbox=self.bbox,
                                    ra=ra_reference, dec=dec_reference, sky_rotation=0.0)
        self.edge_dist = None
        self.kernel_radius = None
        if psf is not None:
            self.load_psf(psf, **kwargs)
        self.mask = None
        self.source_model = None
        self.bright_model = None
        self.n_star = None
        self.saturation = saturation_value  # in counts
        self.background = background_level  # in counts
        if photons_per_jansky is None:
            photon_energy = constants.Planck*constants.speed_of_light/(bandpass.calc_eff_wavelen()/1e9)
            photons_per_jansky = (1e-26 * (self.photParams.effarea / 1e4) *
                                  bandpass.calc_bandwidth() / photon_energy)

        self.counts_per_jansky = photons_per_jansky / self.photParams.gain

    def load_psf(self, psf, edge_dist=None, _kernel_radius=None, **kwargs):
        """Load a PSF class from galsim. The class needs to have two methods, getFWHM() and drawImage()."""
        """
        @param edge_dist: Number of pixels from the edge of the image to exclude sources. May be negative.
        @param kernel_radius: radius in pixels to use when gridding sources in fast_dft.py.
                              Best to be calculated from psf, unless you know what you are doing!
        """
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2. * np.log(2)))
        self.psf = psf
        CoordsXY = self.coord
        kernel_min_radius = np.ceil(5 * psf.getFWHM() * fwhm_to_sigma / CoordsXY.scale())
        self.kernel_radius = _kernel_radius
        if self.kernel_radius < kernel_min_radius:
            self.kernel_radius = kernel_min_radius
        if self.edge_dist is None:
            if CoordsXY.pad > 1:
                self.edge_dist = 0
            else:
                self.edge_dist = 5 * psf.getFWHM() * fwhm_to_sigma / CoordsXY.scale()

    def load_catalog(self, name=None, sed_list=None, n_star=None, seed=None, **kwargs):
        """Load or generate a catalog of stars to be used for the simulations."""
        """
        @param name: name of flux entry to use for catalg. Only important for external use of the catalog.
        @param n_star: number of stars to include in the simulated catalog.
        @param seed: random number generator seed value. Allows simulations to be recreated.
        """
        bright_sigma_threshold = 3.0  # Stars with flux this many sigma over the mean are 'bright'
        bright_flux_threshold = 0.1  # To be 'bright' a star's flux must exceed
        #                              this fraction of the flux of all faint stars
        CoordsXY = self.coord
        x_size_use = CoordsXY.xsize(base=True)  # / 2.0
        y_size_use = CoordsXY.ysize(base=True)  # / 2.0
        if self.edge_dist is not None:
            x_size_use += self.edge_dist
            y_size_use += self.edge_dist
        sky_radius = np.sqrt(x_size_use**2.0 + y_size_use**2.0) * self.wcs.pixelScale().asDegrees()
        self.catalog = _cat_sim(sky_radius=sky_radius, wcs=self.wcs, _wcs_ref=self._wcs_ref,
                                name=name, n_star=n_star, seed=seed, **kwargs)
        self.seed = seed
        if sed_list is None:
            sed_list = self.sed_list

        # if catalog.isContiguous()
        xv_full = self.catalog.getX()
        yv_full = self.catalog.getY()

        # The catalog may include stars outside of the field of view of this observation, so trim those.
        xmin = self.edge_dist
        xmax = CoordsXY.xsize(base=True) - self.edge_dist
        ymin = self.edge_dist
        ymax = CoordsXY.ysize(base=True) - self.edge_dist
        include_flag = ((xv_full >= xmin) & (xv_full < xmax) & (yv_full >= ymin) & (yv_full < ymax))
        xv = xv_full[include_flag]
        yv = yv_full[include_flag]
        catalog_use = self.catalog[include_flag]

        n_star = len(xv)
        flux_arr = np.zeros((n_star, self.n_step))

        for _i, source_record in enumerate(catalog_use):
            star_spectrum = _star_gen(sed_list=sed_list, bandpass=self.bandpass, source_record=source_record,
                                      bandpass_highres=self.bandpass_highres, photParams=self.photParams)
            flux_arr[_i, :] = np.array([flux_val for flux_val in star_spectrum])
        flux_tot = np.sum(flux_arr, axis=1)
        if n_star > 3:
            cat_sigma = np.std(flux_tot[flux_tot - np.median(flux_tot) <
                                        bright_sigma_threshold * np.std(flux_tot)])
            bright_inds = (np.where(flux_tot - np.median(flux_tot) > bright_sigma_threshold * cat_sigma))[0]
            if len(bright_inds) > 0:
                flux_faint = np.sum(flux_arr) - np.sum(flux_tot[bright_inds])
                bright_inds = [i_b for i_b in bright_inds
                               if flux_tot[i_b] > bright_flux_threshold * flux_faint]
            bright_flag = np.zeros(n_star)
            bright_flag[bright_inds] = 1
        else:
            bright_flag = np.ones(n_star)

        self.n_star = n_star
        self.star_flux = flux_arr
        CoordsXY.set_flag(bright_flag == 1)
        CoordsXY.set_x(xv)
        CoordsXY.set_y(yv)

    def make_reference_catalog(self, output_directory=None, filter_list=None, magnitude_limit=None):
        """Create a reference catalog and write it to disk."""
        """
        @param output_directory: path to directory where catalog will be saved.
        @param filter_list: list of filters to use to create magnitudes
        @param magnitude_limit: faintest magnitude star to include in the catalog
        """
        flux_to_jansky = 1.0e26
        if filter_list is None:
            filter_list = ['u', 'g', 'r', 'i', 'z', 'y']
        n_filter = len(filter_list)
        wavelength_step = self.bandpass.wavelen_step
        bp_midrange = _load_bandpass(band_name=filter_list[n_filter // 2], wavelength_step=wavelength_step)
        bandwidth_hz = bp_midrange.calc_bandwidth()
        if self.catalog is None:
            raise ValueError("You must first load a catalog with load_catalog!")
        schema = self.catalog.getSchema()
        schema_entry = schema.extract("*_fluxRaw", ordered='true')
        fluxName = schema_entry.iterkeys().next()
        flux_raw = self.catalog[schema.find(fluxName).key]
        magnitude_raw = -2.512 * np.log10(flux_raw * flux_to_jansky / bandwidth_hz / 3631.0)
        if magnitude_limit is None:
            magnitude_limit = (np.min(magnitude_raw) + np.max(magnitude_raw)) / 2.0
        src_use = magnitude_raw < magnitude_limit

        n_star = np.sum(src_use)
        print("Writing %i stars brighter than %2.1f mag to reference catalog in %i bands"
              % (n_star, magnitude_limit, n_filter))
        print("Min/max magnitude: ", np.min(magnitude_raw), np.max(magnitude_raw))
        schema = self.catalog.getSchema()
        raKey = schema.find("coord_ra").key
        decKey = schema.find("coord_dec").key
        data_array = np.zeros((n_star, 3 + n_filter))
        data_array[:, 0] = np.arange(n_star)
        data_array[:, 1] = np.degrees((self.catalog[raKey])[src_use])
        data_array[:, 2] = np.degrees((self.catalog[decKey])[src_use])
        header = "uniqueId, raJ2000, decJ2000"

        for _f, f_name in enumerate(filter_list):
            bp = _load_bandpass(band_name=f_name, wavelength_step=wavelength_step)
            bp_highres = _load_bandpass(band_name=f_name, wavelength_step=None)
            # build an empty list to store the flux of each star in each filter
            mag_single = []
            for _i, source_rec in enumerate(self.catalog):
                if src_use[_i]:
                    star_spectrum = _star_gen(sed_list=self.sed_list, bandpass=bp, source_record=source_rec,
                                              bandpass_highres=bp_highres, photParams=self.photParams)
                    magnitude = -2.512*np.log10(np.sum(star_spectrum)*self.counts_per_jansky/3631.0)
                    mag_single.append(magnitude)
            data_array[:, 3 + _f] = np.array(mag_single)
            header += ", lsst_%s" % f_name

        base_filename = "starfast_ref_obs_s%in%i.txt" % (self.seed, n_star)
        if output_directory is None:
            file_path = base_filename
        else:
            if output_directory[-4:] == ".txt":
                file_path = output_directory
            else:
                file_path = os.path.join(output_directory, base_filename)
        np.savetxt(file_path, data_array, delimiter=", ", header=header)

    def simulate(self, verbose=True, **kwargs):
        """Call fast_dft.py to construct the input sky model for each frequency slice prior to convolution."""
        CoordsXY = self.coord
        n_bright = CoordsXY.n_flag()
        n_faint = self.n_star - n_bright
        if verbose:
            print("Simulating %i stars within observable region" % self.n_star)
        bright_flag = True
        if n_faint > 0:
            CoordsXY.set_oversample(1)
            flux = self.star_flux[CoordsXY.flag != bright_flag]
            timing_model = -time.time()
            self.source_model = fast_dft(flux, CoordsXY.x_loc(bright=False), CoordsXY.y_loc(bright=False),
                                         x_size=CoordsXY.xsize(), y_size=CoordsXY.ysize(),
                                         kernel_radius=self.kernel_radius, no_fft=False, **kwargs)
            timing_model += time.time()
            if verbose:
                print(_timing_report(n_star=n_faint, bright=False, timing=timing_model))
        if n_bright > 0:
            CoordsXY.set_oversample(2)
            flux = self.star_flux[CoordsXY.flag == bright_flag]
            timing_model = -time.time()
            self.bright_model = fast_dft(flux, CoordsXY.x_loc(bright=True), CoordsXY.y_loc(bright=True),
                                         x_size=CoordsXY.xsize(), y_size=CoordsXY.ysize(),
                                         kernel_radius=CoordsXY.xsize(), no_fft=False, **kwargs)
            timing_model += time.time()
            if verbose:
                print(_timing_report(n_star=n_bright, bright=True, timing=timing_model))

    def convolve(self, seed=None, sky_noise=0, instrument_noise=0, photon_noise=0, verbose=True,
                 elevation=None, azimuth=None, exposureId=None, **kwargs):
        """Convolve a simulated sky with a given PSF. Returns an LSST exposure.

        @param exposureId: unique identificatin number for different runs of the same simulation.
        """
        CoordsXY = self.coord
        sky_noise_gen = _sky_noise_gen(CoordsXY, seed=seed, amplitude=sky_noise,
                                       n_step=self.n_step, verbose=verbose)
        if self.source_model is not None:
            source_image = self._convolve_subroutine(sky_noise_gen, verbose=verbose, bright=False,
                                                     elevation=elevation, azimuth=azimuth, **kwargs)
        else:
            source_image = 0.0
        if self.bright_model is not None:
            bright_image = self._convolve_subroutine(sky_noise_gen, verbose=verbose, bright=True,
                                                     elevation=elevation, azimuth=azimuth, **kwargs)
        else:
            bright_image = 0.0
        return_image = (source_image + bright_image) + self.background
        variance = return_image[:, :]

        if photon_noise > 0:
            rand_gen = np.random
            if seed is not None:
                rand_gen.seed(seed - 1.2)
            photon_scale = self.photParams.gain/photon_noise
            return_image = np.round(rand_gen.poisson(variance*photon_scale)/photon_scale)
        if instrument_noise is None:
            instrument_noise = self.photParams.readnoise

        if instrument_noise > 0:
            rand_gen = np.random
            variance += instrument_noise
            if seed is not None:
                rand_gen.seed(seed - 1.1)
            noise_image = rand_gen.normal(scale=instrument_noise, size=return_image.shape)
            return_image += noise_image
        exposure = self.create_exposure(return_image, variance=variance, boresightRotAngle=self.sky_rotation,
                                        ra=self.ra, dec=self.dec, elevation=elevation, azimuth=azimuth,
                                        exposureId=exposureId, **kwargs)
        return(exposure)

    def _convolve_subroutine(self, sky_noise_gen, psf=None, verbose=True, bright=False,
                             elevation=None, azimuth=None, **kwargs):
        CoordsXY = self.coord
        if bright:
            CoordsXY.set_oversample(2)
        else:
            CoordsXY.set_oversample(1)
        dcr_gen = _dcr_generator(self.bandpass, pixel_scale=CoordsXY.scale(),
                                 elevation=elevation, azimuth=azimuth + self.sky_rotation, **kwargs)
        convol = np.zeros((CoordsXY.ysize(), CoordsXY.xsize() // 2 + 1), dtype='complex64')
        if psf is None:
            psf = self.psf
        if self.psf is None:
            self.load_psf(psf)
        psf_norm = 1.0 / self.psf.getFlux()
        timing_fft = -time.time()

        for _i, offset in enumerate(dcr_gen):
            if bright:
                source_model_use = self.bright_model[_i]
            else:
                source_model_use = self.source_model[_i]

            psf_image = psf.drawImage(scale=CoordsXY.scale(), method='fft', offset=offset,
                                      nx=CoordsXY.xsize(), ny=CoordsXY.ysize(), use_true_center=False)
            try:
                #  Note: if adding sky noise, it should only added once (check if the generator is exhausted)
                source_model_use += next(sky_noise_gen)
            except StopIteration:
                pass
            convol_single = source_model_use * rfft2(psf_image.array)
            convol += convol_single
        return_image = np.real(fftshift(irfft2(convol))) * CoordsXY.oversample**2.0 * psf_norm
        timing_fft += time.time()
        if verbose:
            print("FFT timing for %i DCR planes: [%0.3fs | %0.3fs per plane]"
                  % (self.n_step, timing_fft, timing_fft / self.n_step))
        return_image = return_image[CoordsXY.ymin():CoordsXY.ymax():CoordsXY.oversample,
                                    CoordsXY.xmin():CoordsXY.xmax():CoordsXY.oversample]
        if bright:
            CoordsXY.set_oversample(1)
        return(return_image)

    def create_exposure(self, array, variance=None, elevation=None, azimuth=None, snap=0,
                        exposureId=0, ra=nanAngle, dec=nanAngle, boresightRotAngle=nanAngle, **kwargs):
        """Convert a numpy array to an LSST exposure, and units of electron counts.

        @param array: numpy array to use as the data for the exposure
        @param variance: optional numpy array to use as the variance plane of the exposure.
                         If None, the absoulte value of 'array' is used for the variance plane.
        @param elevation: Elevation angle of the observation, in degrees.
        @param azimuth: Azimuth angle of the observation, in degrees.
        @param snap: snap ID to add to the metadata of the exposure. Required to mimic Phosim output.
        @param exposureId: observation ID of the exposure, a long int.
        @param **kwargs: Any additional keyword arguments will be added to the metadata of the exposure.
        @return Returns an LSST exposure.
        """
        exposure = afwImage.ExposureD(self.bbox)
        exposure.setWcs(self.wcs)
        # We need the filter name in the exposure metadata, and it can't just be set directly
        try:
            exposure.setFilter(afwImage.Filter(self.photParams.bandpass))
        except:
            filterPolicy = pexPolicy.Policy()
            filterPolicy.add("lambdaEff", self.bandpass.calc_eff_wavelen())
            afwImage.Filter.define(afwImage.FilterProperty(self.photParams.bandpass, filterPolicy))
            exposure.setFilter(afwImage.Filter(self.photParams.bandpass))
            # Need to reset afwImage.Filter to prevent an error in future calls to daf_persistence.Butler
            try:
                afwImage.FilterProperty_reset()
            except:
                pass  # Do nothing?
        exposure.setPsf(self._calc_effective_psf(elevation=elevation, azimuth=azimuth, **kwargs))
        exposure.getMaskedImage().getImage().getArray()[:, :] = array
        if variance is None:
            variance = np.abs(array)
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance

        if self.mask is not None:
            exposure.getMaskedImage().getMask().getArray()[:, :] = self.mask

        hour_angle = (90.0 - elevation)*np.cos(np.radians(azimuth))/15.0
        mjd = 59000.0 + (lsst_lat.asDegrees()/15.0 - hour_angle)/24.0
        airmass = 1.0/np.sin(np.radians(elevation))
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))

        meta.add("TAI", mjd)
        meta.add("MJD-OBS", mjd)

        meta.add("EXTTYPE", "IMAGE")
        meta.add("EXPTIME", 30.0)
        meta.add("AIRMASS", airmass)
        meta.add("ZENITH", 90 - elevation)
        meta.add("AZIMUTH", azimuth)
        # Add all additional keyword arguments to the metadata.
        for add_item in kwargs:
            meta.add(add_item, kwargs[add_item])

        visitInfo = afwImage.makeVisitInfo(
            exposureId=int(exposureId),
            exposureTime=30.0,
            darkTime=30.0,
            date=DateTime(mjd),
            ut1=mjd,
            boresightRaDec=IcrsCoord(ra, dec),
            boresightAzAlt=Coord(Angle(np.radians(azimuth)), Angle(np.radians(elevation))),
            boresightAirmass=airmass,
            boresightRotAngle=Angle(np.radians(boresightRotAngle)),
            observatory=Observatory(lsst_lon, lsst_lat, lsst_alt),)
        exposure.getInfo().setVisitInfo(visitInfo)
        return exposure

    def _calc_effective_psf(self, elevation=None, azimuth=None, psf_size=29, **kwargs):
        CoordsXY = self.coord
        dcr_gen = _dcr_generator(self.bandpass, pixel_scale=CoordsXY.scale(),
                                 elevation=elevation, azimuth=azimuth + self.sky_rotation, **kwargs)

        psf_image = afwImage.ImageD(psf_size, psf_size)
        for offset in dcr_gen:
            psf_single = self.psf.drawImage(scale=CoordsXY.scale(), method='fft', offset=offset,
                                            nx=psf_size, ny=psf_size, use_true_center=False)
            psf_image.getArray()[:, :] += psf_single.array
        psfK = afwMath.FixedKernel(psf_image)
        return(measAlg.KernelPsf(psfK))


def _sky_noise_gen(CoordsXY, seed=None, amplitude=None, n_step=1, verbose=False):
    """Generate random sky noise in Fourier space."""
    """
    @param seed: Random number generator seed value. If None, the current system time will be used.
    @param amplitude: Scale factor of random sky noise, in Jy. If None or <=0, no noise is added.
    @param n_step: Number of sub-band planes used for the simulation. Supplied internally, for normalization.
    """
    if amplitude > 0:
        if verbose:
            print("Adding sky noise with amplitude %f" % amplitude)
        rand_gen = np.random
        if seed is not None:
            rand_gen.seed(seed - 1)
        #  Note: it's important to use CoordsXY.xsize() here, since CoordsXY is updated for bright stars
        y_size_use = CoordsXY.ysize()
        x_size_use = CoordsXY.xsize() // 2 + 1
        amplitude_use = amplitude / (np.sqrt(n_step / (x_size_use * y_size_use)))
        for _i in range(n_step):
            rand_fft = (rand_gen.normal(scale=amplitude_use, size=(y_size_use, x_size_use)) +
                        1j * rand_gen.normal(scale=amplitude_use, size=(y_size_use, x_size_use)))
            yield(rand_fft)


class _CoordsXY:
    def __init__(self, pad_image=1.5, pixel_scale=None, x_size=None, y_size=None):
        self._x_size = x_size
        self._y_size = y_size
        self.pixel_scale = pixel_scale
        self.oversample = 1
        self.pad = pad_image
        self.flag = None

    def set_x(self, x_loc):
        self._x = x_loc

    def set_y(self, y_loc):
        self._y = y_loc

    def set_flag(self, flag):
        self.flag = flag.astype(bool)

    def n_flag(self):
        if self.flag is None:
            n = 0
        else:
            n = np.sum(self.flag)
        return(n)

    def set_oversample(self, oversample):
        self.oversample = int(oversample)

    def xsize(self, base=False):
        if base:
            return(int(self._x_size))
        else:
            return(int(self._x_size * self.pad) * self.oversample)

    def xmin(self):
        return(int(self.oversample * (self._x_size * (self.pad - 1) // 2)))

    def xmax(self):
        return(int(self.xmin() + self._x_size * self.oversample))

    def ysize(self, base=False):
        if base:
            return(int(self._y_size))
        else:
            return(int(self._y_size * self.pad) * self.oversample)

    def ymin(self):
        return(int(self.oversample * (self._y_size * (self.pad - 1) // 2)))

    def ymax(self):
        return(int(self.ymin() + self._y_size * self.oversample))

    def scale(self):
        return(self.pixel_scale / self.oversample)

    def x_loc(self, bright=False):
        x_loc = self._x * self.oversample + self.xmin()
        if self.flag is not None:
            x_loc = x_loc[self.flag == bright]
        return(x_loc)

    def y_loc(self, bright=False):
        y_loc = self._y * self.oversample + self.ymin()
        if self.flag is not None:
            y_loc = y_loc[self.flag == bright]
        return(y_loc)


def _create_wcs(bbox=None, pixel_scale=None, ra=None, dec=None, sky_rotation=None):
    """Create a wcs (coordinate system)."""
    crval = IcrsCoord(ra, dec)
    crpix = afwGeom.Box2D(bbox).getCenter()
    cd1_1 = (pixel_scale * afwGeom.arcseconds * np.cos(np.radians(sky_rotation))).asDegrees()
    cd1_2 = (-pixel_scale * afwGeom.arcseconds * np.sin(np.radians(sky_rotation))).asDegrees()
    cd2_1 = (pixel_scale * afwGeom.arcseconds * np.sin(np.radians(sky_rotation))).asDegrees()
    cd2_2 = (pixel_scale * afwGeom.arcseconds * np.cos(np.radians(sky_rotation))).asDegrees()
    return(afwImage.makeWcs(crval, crpix, cd1_1, cd1_2, cd2_1, cd2_2))


def _timing_report(n_star=None, bright=False, timing=None):
    if bright:
        bright_star = "bright "
    else:
        bright_star = ""
    if n_star == 1:
        return("Time to model %i %sstar: [%0.3fs]" % (n_star, bright_star, timing))
    else:
        return("Time to model %i %sstars: [%0.3fs | %0.5fs per star]"
               % (n_star, bright_star, timing, timing / n_star))


def _dcr_generator(bandpass, pixel_scale=None, elevation=50.0, azimuth=0.0, **kwargs):
    """Call the functions that compute Differential Chromatic Refraction (relative to mid-band)."""
    """
    @param bandpass: bandpass object created with load_bandpass
    @param pixel_scale: plate scale in arcsec/pixel
    @param elevation: elevation angle of the center of the image, in decimal degrees.
    @param azimuth: azimuth angle of the observation, in decimal degrees.
    """
    zenith_angle = 90.0 - elevation
    wavelength_midpoint = bandpass.calc_eff_wavelen()
    for wavelength in _wavelength_iterator(bandpass, use_midpoint=True):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the band
        refract_amp = diff_refraction(wavelength=wavelength, wavelength_ref=wavelength_midpoint,
                                      zenith_angle=zenith_angle, **kwargs)
        refract_amp *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
        dx = refract_amp * np.sin(np.radians(azimuth))
        dy = refract_amp * np.cos(np.radians(azimuth))
        yield((dx, dy))


def _cat_sim(seed=None, n_star=None, n_galaxy=None, sky_radius=None, name=None, wcs=None,
             _wcs_ref=None, **kwargs):
    """Wrapper function that generates a semi-realistic catalog of stars."""
    schema = afwTable.SourceTable.makeMinimalSchema()
    if name is None:
        name = "sim"
    fluxName = name + "_fluxRaw"
    flagName = name + "_flag"
    fluxSigmaName = name + "_fluxSigma"
    schema.addField(fluxName, type="D")
    schema.addField(fluxSigmaName, type="D")
    schema.addField(flagName, type="D")
    schema.addField(name + "_Centroid_x", type="D")
    schema.addField(name + "_Centroid_y", type="D")
    schema.addField("temperature", type="D")
    schema.addField("spectral_id", type="D")
    schema.addField("metallicity", type="D")
    schema.addField("gravity", type="D")
    schema.addField("sed", type="D")
    schema.addField("dust", type="D")
    schema.getAliasMap().set('slot_Centroid', name + '_Centroid')

    if _wcs_ref is None:
        _wcs_ref = wcs.clone()
    star_properties = _StellarDistribution(seed=seed, n_star=n_star, sky_radius=sky_radius,
                                           wcs=_wcs_ref, **kwargs)

    catalog = afwTable.SourceCatalog(schema)
    fluxKey = schema.find(fluxName).key
    flagKey = schema.find(flagName).key
    fluxSigmaKey = schema.find(fluxSigmaName).key
    raKey = schema.find("coord_ra").key
    decKey = schema.find("coord_dec").key
    temperatureKey = schema.find("temperature").key
    metalKey = schema.find("metallicity").key
    gravityKey = schema.find("gravity").key
    centroidKey = afwTable.Point2DKey(schema["slot_Centroid"])
    for _i in range(n_star):
        ra = star_properties.ra[_i]
        dec = star_properties.dec[_i]
        source_test_centroid = wcs.skyToPixel(ra, dec)
        source = catalog.addNew()
        source.set(fluxKey, star_properties.raw_flux[_i])
        source.set(centroidKey, source_test_centroid)
        source.set(raKey, ra)
        source.set(decKey, dec)
        source.set(fluxSigmaKey, 0.)
        source.set(temperatureKey, star_properties.temperature[_i])
        source.set(metalKey, star_properties.metallicity[_i])
        source.set(gravityKey, star_properties.surface_gravity[_i])
        source.set(flagKey, False)
    del star_properties
    return(catalog.copy(True))  # Return a copy to make sure it is contiguous in memory.


def _star_gen(sed_list=None, seed=None, bandpass=None, bandpass_highres=None,
              source_record=None, verbose=True, photParams=None):
    """Generate a randomized spectrum at a given temperature over a range of wavelengths."""
    """
        Either use a supplied list of SEDs to be drawn from, or use a blackbody radiation model.
        The output is normalized to sum to the given flux.
        [future] If a seed is supplied, noise can be added to the final spectrum before normalization.

        @param sed_list: Supply a list of SEDs from a call to lsst.sims.photUtils matchStar.
                         If None, a simple blackbody radiation spectrum will be used for each source.
        @param seed: Not yet implemented!
        @param temperature: surface temperature of the star in degrees Kelvin.
        @param metallicity: logarithmic metallicity of the star, relative to solar.
        @param surface_gravity: surface gravity of the star, relative to solar
        @param flux: Total broadband flux of the star, in W/m^2
        @param bandpass: bandpass object created with load_bandpass
        @param bandpass_highres: max resolution bandpass object created with load_bandpass

        @return Returns an array of flux values (one entry per sub-band of the simulation)
    """
    # flux_to_jansky = 1.0e26
    bandwidth_hz = bandpass.calc_bandwidth()
    photon_energy = constants.Planck*constants.speed_of_light/(bandpass.calc_eff_wavelen()/1e9)
    photons_per_jansky = ((photParams.effarea/1e4)*bandwidth_hz/photon_energy)
    flux_to_counts = photons_per_jansky/photParams.gain

    def integral(generator):
        """Simple wrapper to make the math more apparent."""
        return(np.sum(var for var in generator))

    if sed_list is None:
        if verbose:
            print("No sed_list supplied, using blackbody radiation spectra.")
        t_ref = [np.Inf, 0.0]
    else:
        temperatures = np.array([star.temp for star in sed_list])
        t_ref = [temperatures.min(), temperatures.max()]

    schema = source_record.getSchema()
    schema_entry = schema.extract("*_fluxRaw", ordered='true')
    fluxName = schema_entry.iterkeys().next()

    flux_raw = source_record[schema.find(fluxName).key]
    temperature = source_record["temperature"]
    metallicity = source_record["metallicity"]
    surface_gravity = source_record["gravity"]

    # If the desired temperature is outside of the range of models in sed_list, then use a blackbody.
    if temperature >= t_ref[0] and temperature <= t_ref[1]:
        temp_weight = np.abs(temperatures/temperature - 1.0)
        temp_thresh = np.min(temp_weight)
        t_inds = np.where(temp_weight <= temp_thresh)
        t_inds = t_inds[0]  # unpack tuple from np.where()
        n_inds = len(t_inds)
        if n_inds > 1:
            grav_list = np.array([sed_list[_i].logg for _i in t_inds])
            metal_list = np.array([sed_list[_i].logZ for _i in t_inds])
            offset = 10.0  # Add an offset to the values to prevent dividing by zero
            grav_weight = ((grav_list + offset) / (surface_gravity + offset) - 1.0)**2.0
            metal_weight = ((metal_list + offset) / (metallicity + offset) - 1.0)**2.0
            composite_weight = grav_weight + metal_weight
            sed = sed_list[t_inds[np.argmin(composite_weight)]]
        else:
            sed = sed_list[t_inds[0]]

        sb_vals = bandpass_highres.sb.copy()
        bp_use = deepcopy(bandpass_highres)
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            bp_use.sb[:] = 0.
            wl_inds = (bp_use.wavelen >= wave_start) & (bp_use.wavelen < wave_end)
            bp_use.sb[wl_inds] = sb_vals[wl_inds]
            yield sed.calcADU(bp_use, photParams)*flux_raw

    else:
        bp_wavelen, bandpass_vals = bandpass.getBandpass()
        bandpass_gen = (bp for bp in bandpass_vals)
        bandpass_gen2 = (bp2 for bp2 in bandpass_vals)

        h = constants.Planck
        kb = constants.Boltzmann
        c = constants.speed_of_light

        prefactor = 2.0 * (kb * temperature)**4. / (h**3 * c**2)

        def radiance_expansion(x, nterms):
            for n in range(1, nterms + 1):
                poly_term = x**3 / n + 3 * x**2 / n**2 + 6 * x / n**3 + 6 / n**4
                exp_term = np.exp(-n * x)
                yield(poly_term * exp_term)

        def radiance_calc(wavelength_start, wavelength_end, temperature=temperature, nterms=3):
            nu1 = c / (wavelength_start / 1E9)
            nu2 = c / (wavelength_end / 1E9)
            x1 = h * nu1 / (kb * temperature)
            x2 = h * nu2 / (kb * temperature)
            radiance1 = radiance_expansion(x1, nterms)
            radiance2 = radiance_expansion(x2, nterms)
            radiance_integral1 = prefactor * integral(radiance1)
            radiance_integral2 = prefactor * integral(radiance2)
            return(radiance_integral1 - radiance_integral2)

        # integral over the full sed, to convert from W/m**2 to W/m**2/Hz
        radiance_full_integral = radiance_calc(bandpass.wavelen_min / 100.0, bandpass.wavelen_max * 100.0)
        flux_band_fraction = radiance_calc(bandpass.wavelen_min, bandpass.wavelen_max)
        flux_band_fraction /= radiance_full_integral

        radiance_band_integral = 0.0
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            radiance_band_integral += next(bandpass_gen2) * radiance_calc(wave_start, wave_end)
        flux_band_norm = flux_to_counts * flux_raw * flux_band_fraction / bandwidth_hz

        for wave_start, wave_end in _wavelength_iterator(bandpass):
            yield(flux_band_norm * next(bandpass_gen) *
                  radiance_calc(wave_start, wave_end) / radiance_band_integral)


def _load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                   use_filter=True, use_detector=True, highres=False, **kwargs):
    """Load in Bandpass object from sims_photUtils."""
    """
    @param band_name: Common name of the filter used. For LSST, use u, g, r, i, z, or y
    @param wavelength_step: Wavelength resolution, also the wavelength range of each sub-band plane
    @param use_mirror: Flag, include mirror in filter throughput calculation?
    @param use_lens: Flag, use LSST lens in filter throughput calculation?
    @param use_atmos: Flag, use standard atmosphere transmission in filter throughput calculation?
    @param use_filter: Flag, use LSST filters in filter throughput calculation?
    """
    class BandpassMod(Bandpass):
        """Customize a few methods of the Bandpass class from sims_photUtils."""

        def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
            """Calculate effective wavelengths for filters."""
            # This is useful for summary numbers for filters.
            # Calculate effective wavelength of filters.
            if self.phi is None:
                self.sbTophi()
            if wavelength_min is None:
                wavelength_min = np.min(self.wavelen)
            if wavelength_max is None:
                wavelength_max = np.max(self.wavelen)
            w_inds = (self.wavelen >= wavelength_min) & (self.wavelen <= wavelength_max)
            effwavelenphi = (self.wavelen[w_inds] * self.phi[w_inds]).sum() / self.phi[w_inds].sum()
            return effwavelenphi

        def calc_bandwidth(self):
            f0 = constants.speed_of_light / (self.wavelen_min * 1.0e-9)
            f1 = constants.speed_of_light / (self.wavelen_max * 1.0e-9)
            f_cen = constants.speed_of_light / (self.calc_eff_wavelen() * 1.0e-9)
            return(f_cen * 2.0 * (f0 - f1) / (f0 + f1))

    """
    Define the wavelength range and resolution for a given ugrizy band.
    These are defined in case the LSST filter throughputs are not used.
    """
    if highres:
        wavelength_step = None
    band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                 'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
    band_range = band_dict[band_name]
    bandpass = BandpassMod(wavelen_min=band_range[0], wavelen_max=band_range[1],
                           wavelen_step=wavelength_step)
    throughput_dir = getPackageDir('throughputs')
    lens_list = ['baseline/lens1.dat', 'baseline/lens2.dat', 'baseline/lens3.dat']
    mirror_list = ['baseline/m1.dat', 'baseline/m2.dat', 'baseline/m3.dat']
    atmos_list = ['atmos/atmos_11.dat']
    detector_list = ['baseline/detector.dat']
    filter_list = ['baseline/filter_' + band_name + '.dat']
    component_list = []
    if use_mirror:
        component_list += mirror_list
    if use_lens:
        component_list += lens_list
    if use_atmos:
        component_list += atmos_list
    if use_detector:
        component_list += detector_list
    if use_filter:
        component_list += filter_list
    bandpass.readThroughputList(rootDir=throughput_dir, componentList=component_list)
    # Calculate bandpass phi value if required.
    if bandpass.phi is None:
        bandpass.sbTophi()
    return(bandpass)


def _wavelength_iterator(bandpass, use_midpoint=False):
    """Define iterator to ensure that loops over wavelength are consistent."""
    wave_start = bandpass.wavelen_min
    while wave_start < bandpass.wavelen_max:
        wave_end = wave_start + bandpass.wavelen_step
        if wave_end > bandpass.wavelen_max:
            wave_end = bandpass.wavelen_max
        if use_midpoint:
            yield(bandpass.calc_eff_wavelen(wavelength_min=wave_start, wavelength_max=wave_end))
        else:
            yield((wave_start, wave_end))
        wave_start = wave_end


class StarCatalog:
    """A container defining the property ranges of all types of stellar objects used in a simulation."""

    def __init__(self, hottest_star='A', coolest_star='M'):
        """Define the ranges for each type of star."""
        self.star_types = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
        hot_val = self.star_types[hottest_star]
        cool_val = self.star_types[coolest_star]
        # relative abundance.
        self.abundance = OrderedDict([('M', 76.45), ('K', 12.1), ('G', 7.6), ('F', 3.0),
                                      ('A', 0.6), ('B', 0.13), ('O', 3e-05)])
        for key, val in self.star_types.items():
            if val < cool_val:
                self.abundance[key] = 0.0
            if val > hot_val:
                self.abundance[key] = 0.0

        # Intrinsic luminosity, relative to solar.
        self.luminosity = {'M': (0.01, 0.08), 'K': (0.08, 0.6), 'G': (0.6, 1.5), 'F': (1.5, 5.0),
                           'A': (5.0, 100.0), 'B': (100.0, 3000.0), 'O': (3000.0, 5000.0)}
        # Surface temperature, in degrees Kelvin.
        self.temperature = {'M': (2400, 3700), 'K': (3700, 5200), 'G': (5200, 6000), 'F': (6000, 7500),
                            'A': (7500, 10000), 'B': (10000, 30000), 'O': (30000, 50000)}
        # (Log) metallicity, relative to solar.
        self.metallicity = {'M': (-3.0, 0.5), 'K': (-3.0, 0.5), 'G': (-3.0, 0.5), 'F': (-3.0, 0.5),
                            'A': (-3.0, 0.5), 'B': (-3.0, 0.5), 'O': (-3.0, 0.5)}
        # Surface gravity, relative to solar.
        self.gravity = {'M': (0.0, 0.5), 'K': (0.0, 1.0), 'G': (0.0, 1.5), 'F': (0.5, 2.0),
                        'A': (1.0, 2.5), 'B': (2.0, 4.0), 'O': (3.0, 5.0)}

    def distribution(self, n_star, rand_gen=np.random):
        """Generate a random distribution of stars."""
        max_prob = np.sum(self.abundance.values())
        star_sort = rand_gen.uniform(0.0, max_prob, n_star)
        star_sort.sort()
        star_prob = np.cumsum(self.abundance.values())
        distribution = OrderedDict()
        ind = 0
        for _i, star in enumerate(self.abundance):
            n_use = len([x for x in star_sort[ind:] if x > 0 and x < star_prob[_i]])
            distribution[star] = n_use
            ind += n_use
        return(distribution)

    def gen_luminosity(self, star_type, rand_gen=np.random, n_star=None):
        """Return a random luminosity (rel. solar) in the defined range for the stellar type."""
        param_range = self.luminosity[star_type]
        param_vals = rand_gen.uniform(param_range[0], param_range[1], size=n_star)
        for val in param_vals:
            yield val

    def gen_temperature(self, star_type, rand_gen=np.random, n_star=None):
        """Return a random temperature in the defined range for the stellar type."""
        param_range = self.temperature[star_type]
        param_vals = rand_gen.uniform(param_range[0], param_range[1], size=n_star)
        for val in param_vals:
            yield val

    def gen_metallicity(self, star_type, rand_gen=np.random, n_star=None):
        """Return a random metallicity (log rel. solar) in the defined range for the stellar type."""
        param_range = self.metallicity[star_type]
        param_vals = rand_gen.uniform(param_range[0], param_range[1], size=n_star)
        for val in param_vals:
            yield val

    def gen_gravity(self, star_type, rand_gen=np.random, n_star=None):
        """Return a random surface gravity (rel. solar) in the defined range for the stellar type."""
        param_range = self.gravity[star_type]
        param_vals = rand_gen.uniform(param_range[0], param_range[1], size=n_star)
        for val in param_vals:
            yield val


class _StellarDistribution():
    def __init__(self, seed=None, n_star=None, hottest_star='A', coolest_star='M',
                 sky_radius=None, wcs=None, verbose=True, **kwargs):
        """Function that attempts to return a realistic distribution of stellar properties."""
        """
        Returns temperature, flux, metallicity, surface gravity
        temperature in units Kelvin
        flux in units W/m**2
        metallicity is logarithmic metallicity relative to solar
        surface gravity relative to solar
        """
        lum_solar = 3.846e26  # Solar luminosity, in Watts
        ly = 9.4607e15  # one light year, in meters
        pi = np.pi
        pixel_scale_degrees = wcs.pixelScale().asDegrees()
        pix_origin_offset = 0.5
        x_center, y_center = wcs.getPixelOrigin()
        x_center += pix_origin_offset
        y_center += pix_origin_offset
        max_star_dist = 10000.0  # light years
        min_star_dist = 100.0  # Assume we're not looking at anything close
        luminosity_to_flux = lum_solar / (4.0 * pi * ly**2.0)
        rand_gen = np.random
        if seed is not None:
            rand_gen.seed(seed)

        StarCat = StarCatalog(hottest_star=hottest_star, coolest_star=coolest_star)
        temperature = []
        flux = []
        metallicity = []
        surface_gravity = []
        flux_star = []
        ra = []
        dec = []

        star_dist = StarCat.distribution(n_star, rand_gen=rand_gen)
        for star_type in star_dist:
            n_use = star_dist[star_type]
            temperature_gen = StarCat.gen_temperature(star_type, rand_gen=rand_gen, n_star=n_use)
            luminosity_gen = StarCat.gen_luminosity(star_type, rand_gen=rand_gen, n_star=n_use)
            metallicity_gen = StarCat.gen_metallicity(star_type, rand_gen=rand_gen, n_star=n_use)
            gravity_gen = StarCat.gen_gravity(star_type, rand_gen=rand_gen, n_star=n_use)
            flux_stars_total = 0.0
            d_min = np.sqrt((rand_gen.uniform(min_star_dist, max_star_dist, size=n_use) ** 2.0 +
                             rand_gen.uniform(min_star_dist, max_star_dist, size=n_use) ** 2.0))
            distance_attenuation = (d_min + rand_gen.uniform(0, max_star_dist - min_star_dist, size=n_use))
            star_radial_dist = np.sqrt((rand_gen.uniform(-sky_radius, sky_radius, size=n_use) ** 2.0 +
                                        rand_gen.uniform(-sky_radius, sky_radius, size=n_use) ** 2.0) / 2.0)
            star_angle = rand_gen.uniform(0.0, 2.0 * pi, size=n_use)
            pseudo_x = x_center + star_radial_dist * np.cos(star_angle) / pixel_scale_degrees
            pseudo_y = y_center + star_radial_dist * np.sin(star_angle) / pixel_scale_degrees

            for _i in range(n_use):
                ra_star, dec_star = wcs.pixelToSky(pseudo_x[_i], pseudo_y[_i]).getPosition()
                ra.append(ra_star * afwGeom.degrees)
                dec.append(dec_star * afwGeom.degrees)
                flux_use = next(luminosity_gen) * luminosity_to_flux / distance_attenuation[_i] ** 2.0
                flux.append(flux_use)
                temperature.append(next(temperature_gen))
                metallicity.append(next(metallicity_gen))
                surface_gravity.append(next(gravity_gen))
                flux_stars_total += flux_use
            flux_star.append(flux_stars_total)
        flux_total = np.sum(flux_star)
        flux_star = [100. * _f / flux_total for _f in flux_star]
        info_string = "Number and flux contribution of stars of each type:\n"
        for _i, star_type in enumerate(star_dist):
            info_string += str(" [%s %i| %0.2f%%]" % (star_type, star_dist[star_type], flux_star[_i]))
        if verbose:
            print(info_string)
        self.temperature = temperature
        self.raw_flux = flux
        self.metallicity = metallicity
        self.surface_gravity = surface_gravity
        self.ra = ra
        self.dec = dec


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class _BasicBandpass:
    """Dummy bandpass object for testing."""

    def __init__(self, band_name='g', wavelength_step=1):
        """Define the wavelength range and resolution for a given ugrizy band."""
        band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                     'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
        band_range = band_dict[band_name]
        self.wavelen_min = band_range[0]
        self.wavelen_max = band_range[1]
        self.wavelen_step = wavelength_step

    def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
        """Mimic the calc_eff_wavelen method of the real bandpass class."""
        if wavelength_min is None:
            wavelength_min = self.wavelen_min
        if wavelength_max is None:
            wavelength_max = self.wavelen_max
        return((wavelength_min + wavelength_max) / 2.0)

    def calc_bandwidth(self):
        f0 = constants.speed_of_light / (self.wavelen_min * 1.0e-9)
        f1 = constants.speed_of_light / (self.wavelen_max * 1.0e-9)
        f_cen = constants.speed_of_light / (self.calc_eff_wavelen() * 1.0e-9)
        return(f_cen * 2.0 * (f0 - f1) / (f0 + f1))

    def getBandpass(self):
        """Mimic the getBandpass method of the real bandpass class."""
        wl_gen = _wavelength_iterator(self)
        wavelengths = [wl[0] for wl in wl_gen]
        wavelengths += [self.wavelen_max]
        bp_vals = [1] * len(wavelengths)
        return((wavelengths, bp_vals))


class _BasicSED:
    """Dummy SED for testing."""

    def __init__(self, temperature=5600.0, metallicity=0.0, surface_gravity=1.0):
        wavelen_min = 10.0
        wavelen_max = 2000.0
        wavelen_step = 1
        self.temp = temperature
        self.logg = surface_gravity
        self.logZ = metallicity
        self.wavelen = np.arange(wavelen_min, wavelen_max, wavelen_step)
        self.flambda = np.arange(wavelen_min, wavelen_max, wavelen_step) / wavelen_max


class CoordinatesTestCase(lsst.utils.tests.TestCase):
    """Test the simple coordinate transformation class."""

    def setUp(self):
        """Define parameters used by every test."""
        seed = 42
        rand_gen = np.random
        rand_gen.seed(seed)
        self.pixel_scale = 0.25
        self.pad_image = 1.5
        self.x_size = 128
        self.y_size = 128
        self.n_star = 30
        self.n_bright = 10
        self.x_loc = rand_gen.uniform(high=self.x_size, size=self.n_star)
        self.y_loc = rand_gen.uniform(high=self.y_size, size=self.n_star)
        self.flag_array = np.array([False] * self.n_star)
        self.flag_array[:2 * self.n_bright:2] = True
        self.coords = _CoordsXY(pixel_scale=self.pixel_scale, pad_image=self.pad_image,
                                x_size=self.x_size, y_size=self.y_size)

    def tearDown(self):
        """Clean up."""
        del self.coords
        del self.flag_array

    def test_coord_size_normal_scale(self):
        """Make sure everything gets set, and the math is correct."""
        self.assertAlmostEqual(self.pad_image * self.x_size, self.coords.xsize())
        self.assertAlmostEqual(self.pad_image * self.y_size, self.coords.ysize())

    def test_coord_size_no_scale(self):
        """Make sure we can recover input dimensions."""
        self.assertAlmostEqual(self.x_size, self.coords.xsize(base=True))
        self.assertAlmostEqual(self.y_size, self.coords.ysize(base=True))

    def test_coord_size_over_scale(self):
        """Make sure everything gets set, and the math is correct."""
        self.coords.set_oversample(2)
        self.assertAlmostEqual(2 * self.pad_image * self.x_size, self.coords.xsize())
        self.assertAlmostEqual(2 * self.pad_image * self.y_size, self.coords.ysize())

    def test_coord_size_over_scale_nonint(self):
        """Oversampling must only by integer factors."""
        self.coords.set_oversample(2.3)
        self.assertAlmostEqual(2 * self.pad_image * self.x_size, self.coords.xsize())
        self.assertAlmostEqual(2 * self.pad_image * self.y_size, self.coords.ysize())

    def test_coord_pixel_scale_base(self):
        """Make sure everything gets set, and the math is correct."""
        self.assertEqual(self.pixel_scale, self.coords.scale())

    def test_coord_pixel_scale_over(self):
        """Make sure everything gets set, and the math is correct."""
        self.coords.set_oversample(2)
        self.assertEqual(self.pixel_scale / 2, self.coords.scale())

    def test_bright_count(self):
        """Check that the number of locations flagged as bright is correct."""
        self.coords.set_flag(self.flag_array)
        self.assertEqual(self.n_bright, self.coords.n_flag())

    def test_faint_source_locations(self):
        """Check that locations of faint sources are computed correctly, and that flags are correct."""
        CoordsXY = self.coords
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        CoordsXY.set_flag(self.flag_array)
        bright_condition = True
        faint_x = self.x_loc[self.flag_array != bright_condition]
        faint_y = self.y_loc[self.flag_array != bright_condition]
        abs_diff_x = np.sum(np.abs(faint_x + CoordsXY.xmin() - CoordsXY.x_loc()))
        abs_diff_y = np.sum(np.abs(faint_y + CoordsXY.ymin() - CoordsXY.y_loc()))
        self.assertAlmostEqual(abs_diff_x, 0)
        self.assertAlmostEqual(abs_diff_y, 0)

    def test_bright_source_locations(self):
        """Check that locations of bright sources are computed correctly, and that flags are correct."""
        CoordsXY = self.coords
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        CoordsXY.set_flag(self.flag_array)
        CoordsXY.set_oversample(2)
        bright_condition = True
        bright_x = 2 * self.x_loc[self.flag_array == bright_condition]
        bright_y = 2 * self.y_loc[self.flag_array == bright_condition]
        abs_diff_x = np.sum(np.abs(bright_x + CoordsXY.xmin() - CoordsXY.x_loc(bright=True)))
        abs_diff_y = np.sum(np.abs(bright_y + CoordsXY.ymin() - CoordsXY.y_loc(bright=True)))
        self.assertAlmostEqual(abs_diff_x, 0)
        self.assertAlmostEqual(abs_diff_y, 0)

    def test_faint_sources_no_flags(self):
        """If there are no flags, all source locations should always be returned."""
        CoordsXY = self.coords
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        self.assertEqual(len(CoordsXY.x_loc()), self.n_star)
        self.assertEqual(len(CoordsXY.y_loc()), self.n_star)

    def test_bright_sources_no_flags(self):
        """If there are no flags, all source locations should always be returned."""
        CoordsXY = self.coords
        CoordsXY.set_oversample(2)
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        self.assertEqual(len(CoordsXY.x_loc(bright=True)), self.n_star)
        self.assertEqual(len(CoordsXY.y_loc(bright=True)), self.n_star)


class DCRTestCase(lsst.utils.tests.TestCase):
    """Test the the calculations of Differential Chromatic Refraction."""

    def setUp(self):
        """Define parameters used by every test."""
        band_name = 'g'
        wavelength_step = 10.0
        self.pixel_scale = 0.25
        self.bandpass = _load_bandpass(band_name=band_name, wavelength_step=wavelength_step)

    def tearDown(self):
        """Clean up."""
        del self.bandpass

    def test_dcr_generator(self):
        """Check that _dcr_generator returns a generator with n_step iterations, and (0,0) at zenith."""
        azimuth = 0.0
        elevation = 90.0
        zenith_dcr = (0.0, 0.0)
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for _i in range(n_step):
            self.assertAlmostEqual(next(dcr_gen), zenith_dcr)
        with self.assertRaises(StopIteration):
            next(dcr_gen)

    def test_dcr_values(self):
        """Check DCR against pre-computed values."""
        azimuth = 0.0
        elevation = 50.0
        dcr_vals = [1.73959243097, 1.44317957935, 1.1427147535, 0.864107322861, 0.604249563363,
                    0.363170721045, 0.137678490152, -0.0730964797295, -0.270866384702, -0.455135994183,
                    -0.628721688199, -0.791313886049, -0.946883455499, -1.08145326102, -1.16120917137]
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for _i in range(n_step):
            self.assertAlmostEqual(next(dcr_gen)[1], dcr_vals[_i])


class BandpassTestCase(lsst.utils.tests.TestCase):
    """Tests of the interface to Bandpass from lsst.sims.photUtils."""

    def setUp(self):
        """Define parameters used by every test."""
        self.band_name = 'g'
        self.wavelength_step = 10
        self.bandpass = _load_bandpass(band_name=self.band_name, wavelength_step=self.wavelength_step)

    def test_step_bandpass(self):
        """Check that the bandpass has necessary methods, and those return the correct number of values."""
        bp = self.bandpass
        bp_wavelen, bandpass_vals = bp.getBandpass()
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        self.assertEqual(n_step + 1, len(bandpass_vals))


class StarGenTestCase(lsst.utils.tests.TestCase):
    """Test the flux calculation for a single star."""

    def setUp(self):
        """Define parameters used by every test."""
        self.bandpass = _BasicBandpass(band_name='g', wavelength_step=10)
        flux_raw = 1E-9
        temperature = 5600.0
        schema = afwTable.SourceTable.makeMinimalSchema()
        schema.addField("test_fluxRaw", type="D")
        schema.addField("test_Centroid_x", type="D")
        schema.addField("test_Centroid_y", type="D")
        schema.addField("temperature", type="D")
        schema.addField("spectral_id", type="D")
        schema.addField("metallicity", type="D")
        schema.addField("gravity", type="D")
        catalog = afwTable.SourceCatalog(schema)
        source_rec = catalog.addNew()
        source_rec.set("test_fluxRaw", flux_raw)
        source_rec.set("temperature", temperature)
        self.source_rec = source_rec

    def tearDown(self):
        """Clean up."""
        del self.bandpass
        del self.source_rec

    def test_blackbody_spectrum(self):
        """Check the blackbody spectrum against pre-computed values."""
        star_gen = _star_gen(source_record=self.source_rec, bandpass=self.bandpass, verbose=False)
        spectrum = np.array([flux for flux in star_gen])
        pre_comp_spectrum = np.array([5.763797967, 5.933545118, 6.083468705, 6.213969661,
                                      6.325613049, 6.419094277, 6.495208932, 6.554826236,
                                      6.598866015, 6.628278971, 6.644030031, 6.647084472,
                                      6.638396542, 6.618900302, 4.616292143])
        abs_diff_spectrum = np.sum(np.abs(spectrum - pre_comp_spectrum))
        self.assertAlmostEqual(abs_diff_spectrum, 0.0)

    def test_sed_spectrum(self):
        """Check a spectrum defined by an SED against pre-computed values."""
        sed_list = [_BasicSED(self.source_rec["temperature"])]
        star_gen = _star_gen(sed_list=sed_list, source_record=self.source_rec,
                             bandpass=self.bandpass, verbose=True)
        spectrum = np.array([flux for flux in star_gen])
        pre_comp_spectrum = np.array([1.06433106, 1.09032205, 1.11631304, 1.14230403, 1.16829502,
                                      1.19428601, 1.22027700, 1.24626799, 1.27225898, 1.29824997,
                                      1.32424096, 1.35023195, 1.37622294, 1.40221393, 0.99701439])
        abs_diff_spectrum = np.sum(np.abs(spectrum - pre_comp_spectrum))
        self.assertAlmostEqual(abs_diff_spectrum, 0.0)


class StellarDistributionTestCase(lsst.utils.tests.TestCase):
    """Verify that the random catalog generation is unchanged."""

    def setUp(self):
        """Define parameters used by every test."""
        self.x_size = 10
        self.y_size = 10
        pixel_scale = 0.25
        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.ExtentI(self.x_size, self.y_size))
        self.seed = 42
        self.wcs = _create_wcs(pixel_scale=pixel_scale, bbox=bbox,
                               ra=lsst_lon, dec=lsst_lat, sky_rotation=0)
        pixel_radius = np.sqrt(((self.x_size / 2.0)**2.0 + (self.y_size / 2.0)**2.0) / 2.0)
        self.sky_radius = pixel_radius * self.wcs.pixelScale().asDegrees()

    def tearDown(self):
        """Clean up."""
        del self.wcs

    def test_star_type_properties(self):
        """Check that the properties of stars of a given type all fall in the right ranges."""
        star_properties = _StellarDistribution(seed=self.seed, wcs=self.wcs, sky_radius=self.sky_radius,
                                               n_star=3, hottest_star='G', coolest_star='G', verbose=False)
        temperature = star_properties.temperature
        metallicity = star_properties.metallicity
        surface_gravity = star_properties.surface_gravity
        temp_range_g_star = [5200, 6000]
        grav_range_g_star = [0.0, 1.5]
        metal_range_g_star = [-3.0, 0.5]
        self.assertLessEqual(np.max(temperature), temp_range_g_star[1])
        self.assertGreaterEqual(np.min(temperature), temp_range_g_star[0])

        self.assertLessEqual(np.max(surface_gravity), grav_range_g_star[1])
        self.assertGreaterEqual(np.min(surface_gravity), grav_range_g_star[0])

        self.assertLessEqual(np.max(metallicity), metal_range_g_star[1])
        self.assertGreaterEqual(np.min(metallicity), metal_range_g_star[0])

    def test_star_xy_range(self):
        """Check that star pixel coordinates are all in range."""
        star_properties = _StellarDistribution(seed=self.seed, wcs=self.wcs, sky_radius=self.sky_radius,
                                               n_star=3, hottest_star='G', coolest_star='G', verbose=False)
        ra = star_properties.ra
        dec = star_properties.dec
        x = []
        y = []
        for _ra, _dec in zip(ra, dec):
            _x, _y = self.wcs.skyToPixel(_ra, _dec)
            x.append(_x)
            y.append(_y)
        self.assertLess(np.max(x), self.x_size)
        self.assertGreaterEqual(np.min(x), 0.0)

        self.assertLess(np.max(y), self.y_size)
        self.assertGreaterEqual(np.min(y), 0.0)


class SkyNoiseTestCase(lsst.utils.tests.TestCase):
    """Verify that the random catalog generation is unchanged."""

    def setUp(self):
        """Define parameters used by every test."""
        self.coord = _CoordsXY(pad_image=1, x_size=64, y_size=64)
        self.n_step = 3
        self.amplitude = 1.0
        self.seed = 3

    def tearDown(self):
        """Clean up."""
        del self.coord

    def test_noise_sigma(self):
        """The sky noise should be normalized such that the standard deviation of the image == amplitude."""
        CoordsXY = self.coord
        noise_gen = _sky_noise_gen(CoordsXY, seed=self.seed, amplitude=self.amplitude,
                                   n_step=self.n_step, verbose=False)
        noise_fft = next(noise_gen)
        for fft_single in noise_gen:
            noise_fft += fft_single
        noise_image = np.real(fftshift(irfft2(noise_fft)))
        dimension = np.sqrt(CoordsXY.xsize() * CoordsXY.ysize())
        self.assertLess(np.abs(np.std(noise_image) - self.amplitude), 1.0 / dimension)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Temp."""

    pass


def setup_module(module):
    """Temp."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
