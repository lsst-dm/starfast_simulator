# LSST Data Management System
# Copyright 2016-2018 University of Washington
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

import astropy.table
from astropy.coordinates import ICRS
from astropy import units
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
import os
from scipy import constants
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.coord import Observatory, Weather
from lsst.afw.coord.refraction import differentialRefraction
from lsst.geom import Angle, degrees, arcseconds
import lsst.geom as geom
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
from lsst.daf.base import DateTime
import lsst.meas.algorithms as measAlg
from lsst.sims.catUtils.matchSED.matchUtils import matchStar
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.utils import getPackageDir
from fast_dft import fast_dft
import time

nanFloat = float("nan")
nanAngle = Angle(nanFloat)


lsst_lat = -30.244639*degrees
lsst_lon = -70.749417*degrees
lsst_alt = 2663.
lsst_temperature = 20.*units.Celsius  # in degrees Celcius
lsst_humidity = 40.  # in percent
lsst_pressure = 73892.*units.pascal

lsst_weather = Weather(lsst_temperature.value, lsst_pressure.value, lsst_humidity)
lsst_observatory = Observatory(lsst_lon, lsst_lat, lsst_alt)


class StarSim:
    """Class that defines a random simulated region of sky, and allows fast transformations."""

    def __init__(self, psf=None, pixel_scale=None, pad_image=1.5, catalog=None, sed_list=None,
                 x_size=512, y_size=512, band_name='g', photons_per_jansky=None,
                 ra=None, dec=None, ra_reference=None, dec_reference=None,
                 sky_rotation=0.0, exposure_time=30.0, saturation_value=65000,
                 background_level=314, attenuation=1.0, quasar_catalog=None,
                 weather=lsst_weather, observatory=lsst_observatory, **kwargs):
        """Set up the fixed parameters of the simulation."""
        """
        @param psf: psf object from Galsim. Needs to have methods calculateFWHM(), drawImage(),
                    and attribute flux.
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
        @param attenuation: Set higher to manually attenuate the flux of the simulated stars
        """
        bandpass = _load_bandpass(band_name=band_name, **kwargs)
        self.n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min)/bandpass.wavelen_step))
        self.bandpass = bandpass
        self.bandpass_highres = _load_bandpass(band_name=band_name, highres=True, **kwargs)
        self.photParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=pixel_scale,
                                                bandpass=band_name)
        self.weather = weather
        self.observatory = observatory
        self.attenuation = attenuation
        self.band_name = band_name
        if sed_list is None:
            # Load in model SEDs
            matchStarObj = matchStar()
            sed_list = matchStarObj.loadKuruczSEDs()
        self.sed_list = sed_list
        self.catalog = catalog
        self.quasar_catalog = quasar_catalog
        self.coord = _CoordsXY(pixel_scale=self.photParams.platescale, pad_image=pad_image,
                               x_size=x_size, y_size=y_size)
        self.quasar_coord = _CoordsXY(pixel_scale=self.photParams.platescale, pad_image=pad_image,
                                      x_size=x_size, y_size=y_size)
        self.bbox = geom.Box2I(geom.Point2I(0, 0), geom.ExtentI(x_size, y_size))
        self.sky_rotation = sky_rotation
        if ra_reference is None:
            ra_reference = observatory.getLongitude()
        if dec_reference is None:
            dec_reference = observatory.getLatitude()
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
        self.n_quasar = None
        self.star_flux = None
        self.quasar_flux = None
        self.saturation = saturation_value  # in counts
        self.background = background_level  # in counts
        if photons_per_jansky is None:
            photon_energy = constants.Planck*constants.speed_of_light/(bandpass.calc_eff_wavelen()/1e9)
            photons_per_jansky = (1e-26*(self.photParams.effarea/1e4) *
                                  bandpass.calc_bandwidth()/photon_energy)

        self.counts_per_jansky = photons_per_jansky/self.photParams.gain

    def load_psf(self, psf, edge_dist=None, _kernel_radius=None, **kwargs):
        """Load a PSF class from galsim."""
        """
        The class needs to have the following methods:
                                                      calculateFWHM()
                                                      drawImage()
                                                      and attribute flux
        @param edge_dist: Number of pixels from the edge of the image to exclude sources. May be negative.
        @param kernel_radius: radius in pixels to use when gridding sources in fast_dft.py.
                              Best to be calculated from psf, unless you know what you are doing!
        """
        fwhm_to_sigma = 1.0/(2.0*np.sqrt(2.*np.log(2)))
        self.psf = psf
        CoordsXY = self.coord
        kernel_min_radius = np.ceil(5*psf.calculateFWHM()*fwhm_to_sigma/CoordsXY.scale())
        self.kernel_radius = _kernel_radius
        if _kernel_radius is None:
            self.kernel_radius = kernel_min_radius
        elif self.kernel_radius < kernel_min_radius:
            self.kernel_radius = kernel_min_radius
        if self.edge_dist is None:
            if CoordsXY.pad > 1:
                self.edge_dist = 0
            else:
                self.edge_dist = 5*psf.calculateFWHM()*fwhm_to_sigma/CoordsXY.scale()

    def load_catalog(self, name=None, sed_list=None, n_star=None, seed=None, sky_radius=None, **kwargs):
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
        if sky_radius is None:
            sky_radius = np.sqrt(x_size_use**2.0 + y_size_use**2.0)*self.wcs.getPixelScale().asDegrees()
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
                                      bandpass_highres=self.bandpass_highres, photParams=self.photParams,
                                      attenuation=self.attenuation, **kwargs)
            flux_arr[_i, :] = np.array([flux_val for flux_val in star_spectrum])
        flux_tot = np.sum(flux_arr, axis=1)
        if n_star > 3:
            cat_sigma = np.std(flux_tot[flux_tot - np.median(flux_tot) <
                                        bright_sigma_threshold*np.std(flux_tot)])
            bright_inds = (np.where(flux_tot - np.median(flux_tot) > bright_sigma_threshold*cat_sigma))[0]
            if len(bright_inds) > 0:
                flux_faint = np.sum(flux_arr) - np.sum(flux_tot[bright_inds])
                bright_inds = [i_b for i_b in bright_inds
                               if flux_tot[i_b] > bright_flux_threshold*flux_faint]
            bright_flag = np.zeros(n_star)
            bright_flag[bright_inds] = 1
        else:
            bright_flag = np.ones(n_star)

        self.n_star = n_star
        self.star_flux = flux_arr
        CoordsXY.set_flag(bright_flag == 1)
        CoordsXY.set_x(xv)
        CoordsXY.set_y(yv)

    def load_quasar_catalog(self, name=None, n_quasar=None, seed=None, sky_radius=None, **kwargs):
        """Load or generate a catalog of stars to be used for the simulations."""
        """
        @param name: name of flux entry to use for catalg. Only important for external use of the catalog.
        @param n_quasar: number of quasars to include in the simulated catalog.
        @param seed: random number generator seed value. Allows simulations to be recreated.
        """
        bright_sigma_threshold = 3.0  # Stars with flux this many sigma over the mean are 'bright'
        bright_flux_threshold = 0.1  # To be 'bright' a star's flux must exceed
        #                              this fraction of the flux of all faint stars
        CoordsXY = self.quasar_coord
        x_size_use = CoordsXY.xsize(base=True)  # / 2.0
        y_size_use = CoordsXY.ysize(base=True)  # / 2.0
        if self.edge_dist is not None:
            x_size_use += self.edge_dist
            y_size_use += self.edge_dist
        if sky_radius is None:
            sky_radius = np.sqrt(x_size_use**2.0 + y_size_use**2.0)*self.wcs.getPixelScale().asDegrees()
        self.quasar_catalog = _quasar_sim(sky_radius=sky_radius, wcs=self.wcs, _wcs_ref=self._wcs_ref,
                                          name=name, n_quasar=n_quasar, seed=seed, **kwargs)
        self.seed = seed

        xv_full = self.quasar_catalog.getX()
        yv_full = self.quasar_catalog.getY()

        # The catalog may include stars outside of the field of view of this observation, so trim those.
        xmin = self.edge_dist
        xmax = CoordsXY.xsize(base=True) - self.edge_dist
        ymin = self.edge_dist
        ymax = CoordsXY.ysize(base=True) - self.edge_dist
        include_flag = ((xv_full >= xmin) & (xv_full < xmax) & (yv_full >= ymin) & (yv_full < ymax))
        xv = xv_full[include_flag]
        yv = yv_full[include_flag]
        catalog_use = self.quasar_catalog[include_flag]

        n_quasar = len(xv)
        flux_arr = np.zeros((n_quasar, self.n_step))

        quasar_sed = Quasar_sed()

        for _i, source_record in enumerate(catalog_use):
            quasar_spectrum = _quasar_gen(source_record=source_record,
                                          bandpass=self.bandpass,
                                          bandpass_highres=self.bandpass_highres,
                                          photParams=self.photParams,
                                          attenuation=self.attenuation,
                                          quasar_sed=quasar_sed,
                                          **kwargs)
            flux_arr[_i, :] = np.array([flux_val for flux_val in quasar_spectrum])
        flux_tot = np.sum(flux_arr, axis=1)
        if n_quasar > 3:
            cat_sigma = np.std(flux_tot[flux_tot - np.median(flux_tot) <
                                        bright_sigma_threshold*np.std(flux_tot)])
            bright_inds = (np.where(flux_tot - np.median(flux_tot) > bright_sigma_threshold*cat_sigma))[0]
            if len(bright_inds) > 0:
                flux_faint = np.sum(flux_arr) - np.sum(flux_tot[bright_inds])
                bright_inds = [i_b for i_b in bright_inds
                               if flux_tot[i_b] > bright_flux_threshold*flux_faint]
            bright_flag = np.zeros(n_quasar)
            bright_flag[bright_inds] = 1
        else:
            bright_flag = np.ones(n_quasar)

        self.n_quasar = n_quasar
        self.quasar_flux = flux_arr
        CoordsXY.set_flag(bright_flag == 1)
        CoordsXY.set_x(xv)
        CoordsXY.set_y(yv)

    def make_reference_catalog(self, output_directory=None, filter_list=None, magnitude_limit=None, **kwargs):
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
        bp_midrange = _load_bandpass(band_name=filter_list[n_filter//2], wavelength_step=wavelength_step)
        bandwidth_hz = bp_midrange.calc_bandwidth()
        if self.catalog is None:
            raise ValueError("You must first load a catalog with load_catalog!")
        schema = self.catalog.getSchema()
        schema_entry = schema.extract("*_fluxRaw", ordered='true')
        fluxName = next(iter(schema_entry.keys()))
        flux_raw = self.catalog[schema.find(fluxName).key]
        magnitude_raw = -2.512*np.log10(flux_raw*flux_to_jansky/bandwidth_hz/3631.0)
        if magnitude_limit is None:
            magnitude_limit = (np.min(magnitude_raw) + np.max(magnitude_raw))/2.0
        src_use = magnitude_raw < magnitude_limit

        n_star = np.sum(src_use)
        print("Writing %i stars brighter than %2.1f mag to reference catalog in %i bands"
              % (n_star, magnitude_limit, n_filter))
        print("Min/max magnitude: ", np.min(magnitude_raw), np.max(magnitude_raw))
        schema = self.catalog.getSchema()
        raKey = schema.find("coord_ra").key
        decKey = schema.find("coord_dec").key
        data_array = np.zeros((n_star, 3 + 2*n_filter))
        data_array[:, 0] = np.arange(n_star)
        data_array[:, 1] = np.degrees((self.catalog[raKey])[src_use])
        data_array[:, 2] = np.degrees((self.catalog[decKey])[src_use])
        header = "uniqueId, raJ2000, decJ2000"
        data_format = ["%i", "%f", "%f"]

        for _f, f_name in enumerate(filter_list):
            bp = _load_bandpass(band_name=f_name, wavelength_step=wavelength_step)
            bp_highres = _load_bandpass(band_name=f_name, wavelength_step=None)
            scale_factor = self.counts_per_jansky/bp.calc_bandwidth()/3631.0
            # build an empty list to store the flux of each star in each filter
            mag_single = []
            mag_err_single = []
            for _i, source_rec in enumerate(self.catalog):
                if src_use[_i]:
                    star_spectrum = _star_gen(sed_list=self.sed_list, bandpass=bp, source_record=source_rec,
                                              bandpass_highres=bp_highres, photParams=self.photParams,
                                              attenuation=self.attenuation, **kwargs)
                    magnitude = -2.512*np.log10(np.sum(star_spectrum)*scale_factor)
                    magnitude_err = 0.2
                    mag_single.append(magnitude)
                    mag_err_single.append(magnitude_err)
            data_array[:, 3 + 2*_f] = np.array(mag_single)
            data_array[:, 4 + 2*_f] = np.array(mag_err_single)
            header += ", %s" % f_name
            header += ", %s_err" % f_name
            data_format.append("%f")
            data_format.append("%f")

        base_filename = "starfast_ref_obs_s%in%i.txt" % (self.seed, n_star)
        if output_directory is None:
            file_path = base_filename
        else:
            if output_directory[-4:] == ".txt":
                file_path = output_directory
            else:
                file_path = os.path.join(output_directory, base_filename)

        np.savetxt(file_path, data_array, delimiter=", ", header=header, fmt=data_format)

    def simulate(self, verbose=True, useQuasars=False, **kwargs):
        """Call fast_dft.py to construct the input sky model for each frequency slice prior to convolution."""
        if useQuasars:
            CoordsXY = self.quasar_coord
            n_bright = CoordsXY.n_flag()
            n_faint = self.n_quasar - n_bright
            source_flux = self.quasar_flux
            if verbose:
                print("Simulating %i quasars within observable region" % self.n_quasar)
        else:
            CoordsXY = self.coord
            n_bright = CoordsXY.n_flag()
            n_faint = self.n_star - n_bright
            source_flux = self.star_flux
            if verbose:
                print("Simulating %i stars within observable region" % self.n_star)

        bright_flag = True
        if n_faint > 0:
            CoordsXY.set_oversample(1)
            if self.source_model is None:
                self.source_model = np.zeros((self.n_step, CoordsXY.xsize(), CoordsXY.ysize()//2 + 1),
                                             dtype=np.complex128)
            flux = source_flux[CoordsXY.flag != bright_flag]
            timing_model = -time.time()
            self.source_model += fast_dft(flux, CoordsXY.x_loc(bright=False), CoordsXY.y_loc(bright=False),
                                          x_size=CoordsXY.xsize(), y_size=CoordsXY.ysize(),
                                          kernel_radius=self.kernel_radius, no_fft=False, **kwargs)
            timing_model += time.time()
            if verbose:
                print(_timing_report(n_star=n_faint, bright=False, timing=timing_model))
        if n_bright > 0:
            CoordsXY.set_oversample(2)
            if self.bright_model is None:
                self.bright_model = np.zeros((self.n_step, CoordsXY.xsize(), CoordsXY.ysize()//2 + 1),
                                             dtype=np.complex128)
            flux = source_flux[CoordsXY.flag == bright_flag]
            timing_model = -time.time()
            self.bright_model += fast_dft(flux, CoordsXY.x_loc(bright=True), CoordsXY.y_loc(bright=True),
                                          x_size=CoordsXY.xsize(), y_size=CoordsXY.ysize(),
                                          kernel_radius=CoordsXY.xsize(), no_fft=False, **kwargs)
            timing_model += time.time()
            if verbose:
                print(_timing_report(n_star=n_bright, bright=True, timing=timing_model))

    def convolve(self, seed=None, sky_noise=0, instrument_noise=0, photon_noise=0, verbose=True,
                 elevation=None, azimuth=None, exposureId=None, psf=None, **kwargs):
        """Convolve a simulated sky with a given PSF. Returns an LSST exposure.

        @param exposureId: unique identificatin number for different runs of the same simulation.
        """
        CoordsXY = self.coord
        sky_noise_gen = _sky_noise_gen(CoordsXY, seed=seed, amplitude=sky_noise,
                                       n_step=self.n_step, verbose=verbose)
        if self.source_model is not None:
            source_image = self._convolve_subroutine(sky_noise_gen, psf=psf, verbose=verbose, bright=False,
                                                     elevation=elevation, azimuth=azimuth)
        else:
            source_image = 0.0
        if self.bright_model is not None:
            bright_image = self._convolve_subroutine(sky_noise_gen, psf=psf, verbose=verbose, bright=True,
                                                     elevation=elevation, azimuth=azimuth)
        else:
            bright_image = 0.0
        return_image = (source_image + bright_image) + self.background
        variance = np.abs(return_image[:, :])

        if photon_noise > 0:
            rand_gen = np.random
            if seed is not None:
                rand_gen.seed(seed - 2)
            photon_scale = self.photParams.gain/photon_noise
            return_image = np.round(rand_gen.poisson(variance*photon_scale)/photon_scale)
        if instrument_noise is None:
            instrument_noise = self.photParams.readnoise

        if instrument_noise > 0:
            rand_gen = np.random
            variance += instrument_noise
            if seed is not None:
                rand_gen.seed(seed - 1)
            noise_image = rand_gen.normal(scale=instrument_noise, size=return_image.shape)
            return_image += noise_image
        exposure = self.create_exposure(return_image, variance=variance, boresightRotAngle=self.sky_rotation,
                                        ra=self.ra, dec=self.dec, elevation=elevation, azimuth=azimuth,
                                        exposureId=exposureId, **kwargs)
        return(exposure)

    def _convolve_subroutine(self, sky_noise_gen, psf=None, verbose=True, bright=False,
                             elevation=None, azimuth=None):
        CoordsXY = self.coord
        if bright:
            CoordsXY.set_oversample(2)
        else:
            CoordsXY.set_oversample(1)
        dcr_gen = _dcr_generator(self.bandpass, self.weather, self.observatory, pixel_scale=CoordsXY.scale(),
                                 elevation=elevation, azimuth=azimuth + self.sky_rotation)
        convol = np.zeros((CoordsXY.ysize(), CoordsXY.xsize()//2 + 1), dtype='complex64')
        if psf is None:
            psf = self.psf
        if self.psf is None:
            self.load_psf(psf)
        psf_norm = 1.0/self.psf.flux
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
        return_image = np.real(fftshift(irfft2(convol)))*(CoordsXY.oversample**2.0)*psf_norm
        timing_fft += time.time()
        if verbose:
            print("FFT timing for %i DCR planes: [%0.3fs | %0.3fs per plane]"
                  % (self.n_step, timing_fft, timing_fft/self.n_step))
        return_image = return_image[CoordsXY.ymin():CoordsXY.ymax():CoordsXY.oversample,
                                    CoordsXY.xmin():CoordsXY.xmax():CoordsXY.oversample]
        if bright:
            CoordsXY.set_oversample(1)
        return(return_image)

    def create_exposure(self, array, variance=None, elevation=None, azimuth=None, snap=0,
                        exposureId=0, ra=nanAngle, dec=nanAngle, boresightRotAngle=nanFloat, **kwargs):
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
            afwImage.Filter.define(afwImage.FilterProperty(self.photParams.bandpass,
                                                           self.bandpass.calc_eff_wavelen(),
                                                           self.bandpass.wavelen_min,
                                                           self.bandpass.wavelen_max))
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
        # Convert to an astropy coordinate, to use their formatting.
        HA = (ICRS(azimuth*units.degree, 0*units.degree)).ra
        meta.add("HA", HA.to_string(units.hour))  # Convert from degrees to hours.
        meta.add("RA_DEG", ra.asDegrees())
        meta.add("DEC_DEG", dec.asDegrees())
        meta.add("ROTANG", boresightRotAngle)
        meta.add("TEMPERA", self.weather.getAirTemperature())
        meta.add("PRESS", self.weather.getAirPressure())
        # Add all additional keyword arguments to the metadata.
        for add_item in kwargs:
            meta.add(add_item, kwargs[add_item])

        visitInfo = afwImage.makeVisitInfo(
            exposureId=int(exposureId),
            exposureTime=30.0,
            darkTime=30.0,
            date=DateTime(mjd),
            ut1=mjd,
            boresightRaDec=geom.SpherePoint(ra, dec),
            boresightAzAlt=geom.SpherePoint(Angle(np.radians(azimuth)), Angle(np.radians(elevation))),
            boresightAirmass=airmass,
            boresightRotAngle=Angle(np.radians(boresightRotAngle)),
            observatory=self.observatory,
            weather=self.weather)
        exposure.getInfo().setVisitInfo(visitInfo)
        return exposure

    def _calc_effective_psf(self, elevation=None, azimuth=None, psf_size=29, **kwargs):
        CoordsXY = self.coord
        dcr_gen = _dcr_generator(self.bandpass, self.weather, self.observatory, pixel_scale=CoordsXY.scale(),
                                 elevation=elevation, azimuth=azimuth + self.sky_rotation)

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
        x_size_use = CoordsXY.xsize()//2 + 1
        amplitude_use = amplitude/(np.sqrt(n_step/(x_size_use*y_size_use)))
        for _i in range(n_step):
            rand_fft = (rand_gen.normal(scale=amplitude_use, size=(y_size_use, x_size_use)) +
                        1j*rand_gen.normal(scale=amplitude_use, size=(y_size_use, x_size_use)))
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
            return(int(self._x_size*self.pad)*self.oversample)

    def xmin(self):
        return(int(self.oversample*(self._x_size*(self.pad - 1)//2)))

    def xmax(self):
        return(int(self.xmin() + self._x_size*self.oversample))

    def ysize(self, base=False):
        if base:
            return(int(self._y_size))
        else:
            return(int(self._y_size*self.pad)*self.oversample)

    def ymin(self):
        return(int(self.oversample*(self._y_size*(self.pad - 1)//2)))

    def ymax(self):
        return(int(self.ymin() + self._y_size*self.oversample))

    def scale(self):
        return(self.pixel_scale/self.oversample)

    def x_loc(self, bright=False):
        x_loc = self._x*self.oversample + self.xmin()
        if self.flag is not None:
            x_loc = x_loc[self.flag == bright]
        return(x_loc)

    def y_loc(self, bright=False):
        y_loc = self._y*self.oversample + self.ymin()
        if self.flag is not None:
            y_loc = y_loc[self.flag == bright]
        return(y_loc)


def _create_wcs(bbox=None, pixel_scale=None, ra=None, dec=None, sky_rotation=None):
    """Create a wcs (coordinate system)."""
    crval = geom.SpherePoint(ra, dec)
    crpix = geom.Box2D(bbox).getCenter()
    cdMatrix = afwGeom.makeCdMatrix(scale=pixel_scale*arcseconds,
                                    orientation=Angle(sky_rotation),
                                    flipX=True)
    wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)
    return wcs


def _timing_report(n_star=None, bright=False, timing=None):
    if bright:
        bright_star = "bright "
    else:
        bright_star = ""
    if n_star == 1:
        return("Time to model %i %sstar: [%0.3fs]" % (n_star, bright_star, timing))
    else:
        return("Time to model %i %sstars: [%0.3fs | %0.5fs per star]"
               % (n_star, bright_star, timing, timing/n_star))


def _dcr_generator(bandpass, weather, observatory, pixel_scale=None, elevation=50.0, azimuth=0.0):
    """Call the functions that compute Differential Chromatic Refraction (relative to mid-band)."""
    """
    @param bandpass: bandpass object created with load_bandpass
    @param pixel_scale: plate scale in arcsec/pixel
    @param elevation: elevation angle of the center of the image, in decimal degrees.
    @param azimuth: azimuth angle of the observation, in decimal degrees.
    """
    wavelength_midpoint = bandpass.calc_eff_wavelen()
    for wavelength in _wavelength_iterator(bandpass, use_midpoint=True):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the band
        refract_amp = differentialRefraction(wavelength=wavelength, wavelengthRef=wavelength_midpoint,
                                             elevation=Angle(elevation*degrees),
                                             weather=weather, observatory=observatory)
        # Convert refraction amplitude to pixels
        refract_amp = refract_amp.asArcseconds()/pixel_scale
        dx = refract_amp*np.sin(np.radians(azimuth))
        dy = refract_amp*np.cos(np.radians(azimuth))
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
        source_test_centroid = wcs.skyToPixel(geom.SpherePoint(ra, dec))
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
    return(catalog.copy(True))


def _quasar_sim(seed=None, n_quasar=None, sky_radius=None, name=None, wcs=None,
                _wcs_ref=None, **kwargs):
    """Wrapper function that generates a semi-realistic catalog of quasars."""
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
    schema.addField("redshift", type="D")
    schema.getAliasMap().set('slot_Centroid', name + '_Centroid')

    if _wcs_ref is None:
        _wcs_ref = wcs.clone()
    quasar_properties = _QuasarDistribution(seed=seed, n_quasar=n_quasar, sky_radius=sky_radius,
                                            wcs=_wcs_ref, **kwargs)

    catalog = afwTable.SourceCatalog(schema)
    fluxKey = schema.find(fluxName).key
    flagKey = schema.find(flagName).key
    fluxSigmaKey = schema.find(fluxSigmaName).key
    raKey = schema.find("coord_ra").key
    decKey = schema.find("coord_dec").key
    redshiftKey = schema.find("redshift").key
    centroidKey = afwTable.Point2DKey(schema["slot_Centroid"])
    for _i in range(n_quasar):
        ra = quasar_properties.ra[_i]
        dec = quasar_properties.dec[_i]
        source_test_centroid = wcs.skyToPixel(geom.SpherePoint(ra, dec))
        source = catalog.addNew()
        source.set(fluxKey, quasar_properties.raw_flux[_i])
        source.set(centroidKey, source_test_centroid)
        source.set(raKey, ra)
        source.set(decKey, dec)
        source.set(fluxSigmaKey, 0.)
        source.set(redshiftKey, quasar_properties.redshift[_i])
        source.set(flagKey, False)
    del quasar_properties
    return(catalog.copy(True))   # Return a copy to make sure it is contiguous in memory.


class Quasar_sed:
    def __init__(self, sed=None):
        if sed is None:
            quasar_sed_path = os.path.join(os.path.dirname(__file__), 'Vanden_Berk_quasars.txt')
            quasar_sed = astropy.table.Table.read(quasar_sed_path, format='ascii.cds')
        quasar_wl = np.array(quasar_sed['Wave'])  # Wavelength in Angstroms
        base_flux = np.array(quasar_sed['FluxD'])  # Relative flux density
        base_flux /= sum(base_flux)
        self.wavelen = quasar_wl/10.  # Convert to nm
        self.flambda = base_flux

    def calcADU(self, bandpass, photParams, redshift):

        photon_energy = constants.Planck*constants.speed_of_light/(bandpass.calc_eff_wavelen()/1e9)
        photons_per_jansky = (1e-26*(photParams.effarea/1e4) *
                              bandpass.calc_bandwidth()/photon_energy)

        counts_per_jansky = photons_per_jansky/photParams.gain
        counts_per_jansky *= 1.8e13  # 1.8e13 factor to put the result on the same scale as simulated stars
        wl_use = self.wavelen*(1. + redshift)
        bandpass_vals = np.interp(wl_use, bandpass.wavelen, bandpass.sb, 0., 0.)
        simple_adu = np.sum(self.flambda*bandpass_vals)*counts_per_jansky
        return(simple_adu)


def _star_gen(sed_list=None, seed=None, bandpass=None, bandpass_highres=None,
              source_record=None, verbose=True, photParams=None, attenuation=1.0,
              force_flat_spectrum=False, **kwargs):
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
    fluxName = next(iter(schema_entry.keys()))

    flux_raw = source_record[schema.find(fluxName).key]/attenuation
    temperature = source_record["temperature"]
    metallicity = source_record["metallicity"]
    surface_gravity = source_record["gravity"]

    if force_flat_spectrum:
        flux_band_fraction = bandpass.wavelen_step/(bandpass.wavelen_max - bandpass.wavelen_min)
        flux_band_norm = flux_to_counts*flux_raw*flux_band_fraction/bandwidth_hz
        # rough approximation to account for the bandpass only containing a fraction of the full flux
        # without actually performing the full calculation (since this is just a quick debugging option)
        flux_band_norm /= 4.
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            yield(flux_band_norm)
    elif temperature >= t_ref[0] and temperature <= t_ref[1]:
        temp_weight = np.abs(temperatures/temperature - 1.0)
        temp_thresh = np.min(temp_weight)
        t_inds = np.where(temp_weight <= temp_thresh)
        t_inds = t_inds[0]  # unpack tuple from np.where()
        n_inds = len(t_inds)
        if n_inds > 1:
            grav_list = np.array([sed_list[_i].logg for _i in t_inds])
            metal_list = np.array([sed_list[_i].logZ for _i in t_inds])
            offset = 10.0  # Add an offset to the values to prevent dividing by zero
            grav_weight = ((grav_list + offset)/(surface_gravity + offset) - 1.0)**2.0
            metal_weight = ((metal_list + offset)/(metallicity + offset) - 1.0)**2.0
            composite_weight = grav_weight + metal_weight
            sed = sed_list[t_inds[np.argmin(composite_weight)]]
        else:
            sed = sed_list[t_inds[0]]

        sb_vals = bandpass_highres.sb.copy()
        bp_use = deepcopy(bandpass_highres)
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            bp_use.sb = np.zeros_like(bp_use.sb)
            for wl_i, wl in enumerate(bp_use.wavelen):
                bp_use.sb[wl_i] = sb_vals[wl_i] if (wl >= wave_start) & (wl < wave_end) else 0.
            yield sed.calcADU(bp_use, photParams)*flux_raw
    else:
        # If the desired temperature is outside of the range of models in sed_list, then use a blackbody.
        bp_wavelen, bandpass_vals = bandpass.getBandpass()
        bandpass_gen = (bp for bp in bandpass_vals)
        bandpass_gen2 = (bp2 for bp2 in bandpass_vals)

        h = constants.Planck
        kb = constants.Boltzmann
        c = constants.speed_of_light

        prefactor = 2.0*((kb*temperature)**4.)/((h**3)*(c**2))

        def radiance_expansion(x, nterms):
            for n in range(1, nterms + 1):
                poly_term = (x**3)/n + 3*(x**2)/(n**2) + 6*x/(n**3) + 6/(n**4)
                exp_term = np.exp(-n*x)
                yield(poly_term*exp_term)

        def radiance_calc(wavelength_start, wavelength_end, temperature=temperature, nterms=3):
            nu1 = c/(wavelength_start/1E9)
            nu2 = c/(wavelength_end/1E9)
            x1 = h*nu1/(kb*temperature)
            x2 = h*nu2/(kb*temperature)
            radiance1 = radiance_expansion(x1, nterms)
            radiance2 = radiance_expansion(x2, nterms)
            radiance_integral1 = prefactor*integral(radiance1)
            radiance_integral2 = prefactor*integral(radiance2)
            return(radiance_integral1 - radiance_integral2)

        # integral over the full sed, to convert from W/m**2 to W/m**2/Hz
        radiance_full_integral = radiance_calc(bandpass.wavelen_min/100.0, bandpass.wavelen_max*100.0)
        flux_band_fraction = radiance_calc(bandpass.wavelen_min, bandpass.wavelen_max)
        flux_band_fraction /= radiance_full_integral

        radiance_band_integral = 0.0
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            radiance_band_integral += next(bandpass_gen2)*radiance_calc(wave_start, wave_end)
        flux_band_norm = flux_to_counts*flux_raw*flux_band_fraction/bandwidth_hz

        for wave_start, wave_end in _wavelength_iterator(bandpass):
            yield(flux_band_norm*next(bandpass_gen) *
                  radiance_calc(wave_start, wave_end)/radiance_band_integral)


def _quasar_gen(source_record, bandpass, bandpass_highres, photParams, attenuation=1.0, quasar_sed=None):
    if quasar_sed is None:
        quasar_sed = Quasar_sed()

    schema = source_record.getSchema()
    schema_entry = schema.extract("*_fluxRaw", ordered='true')
    fluxName = next(iter(schema_entry.keys()))

    flux_raw = source_record[schema.find(fluxName).key]/attenuation
    redshift = source_record["redshift"]
    sb_vals = bandpass_highres.sb.copy()
    bp_use = deepcopy(bandpass_highres)
    for wave_start, wave_end in _wavelength_iterator(bandpass):
        bp_use.sb = np.zeros_like(bp_use.sb)
        for wl_i, wl in enumerate(bp_use.wavelen):
            bp_use.sb[wl_i] = sb_vals[wl_i] if (wl >= wave_start) & (wl < wave_end) else 0.
        yield quasar_sed.calcADU(bp_use, photParams, redshift)*flux_raw


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
            effwavelenphi = (self.wavelen[w_inds]*self.phi[w_inds]).sum()/self.phi[w_inds].sum()
            return effwavelenphi

        def calc_bandwidth(self):
            f0 = constants.speed_of_light/(self.wavelen_min*1.0e-9)
            f1 = constants.speed_of_light/(self.wavelen_max*1.0e-9)
            f_cen = constants.speed_of_light/(self.calc_eff_wavelen()*1.0e-9)
            return(f_cen*2.0*(f0 - f1)/(f0 + f1))

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
        max_prob = np.sum(list(self.abundance.values()))
        star_sort = rand_gen.uniform(0.0, max_prob, n_star)
        star_sort.sort()
        star_prob = np.cumsum(list(self.abundance.values()))
        distribution = OrderedDict()
        ind = 0
        for _i, star in enumerate(list(self.abundance)):
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


class _StellarDistribution:
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
        pixel_scale_degrees = wcs.getPixelScale().asDegrees()
        pix_origin_offset = 0.5
        x_center, y_center = wcs.getPixelOrigin()
        x_center += pix_origin_offset
        y_center += pix_origin_offset
        max_star_dist = 10000.0  # light years
        min_star_dist = 100.0  # Assume we're not looking at anything close
        luminosity_to_flux = lum_solar/(4.0*pi*(ly**2.0))
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
            d_min = np.sqrt((rand_gen.uniform(min_star_dist, max_star_dist, size=n_use)**2.0 +
                             rand_gen.uniform(min_star_dist, max_star_dist, size=n_use)**2.0))
            distance_attenuation = (d_min + rand_gen.uniform(0, max_star_dist - min_star_dist, size=n_use))
            star_radial_dist = np.sqrt((rand_gen.uniform(-sky_radius, sky_radius, size=n_use)**2.0 +
                                        rand_gen.uniform(-sky_radius, sky_radius, size=n_use)**2.0)/2.0)
            star_angle = rand_gen.uniform(0.0, 2.0*pi, size=n_use)
            pseudo_x = x_center + star_radial_dist*np.cos(star_angle)/pixel_scale_degrees
            pseudo_y = y_center + star_radial_dist*np.sin(star_angle)/pixel_scale_degrees

            for _i in range(n_use):
                ra_star, dec_star = wcs.pixelToSky(geom.Point2D(pseudo_x[_i], pseudo_y[_i]))
                ra.append(ra_star)
                dec.append(dec_star)
                flux_use = next(luminosity_gen)*luminosity_to_flux/distance_attenuation[_i]**2.0
                flux.append(flux_use)
                temperature.append(next(temperature_gen))
                metallicity.append(next(metallicity_gen))
                surface_gravity.append(next(gravity_gen))
                flux_stars_total += flux_use
            flux_star.append(flux_stars_total)
        flux_total = np.sum(flux_star)
        flux_star = [100.*_f / flux_total for _f in flux_star]
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


class _QuasarDistribution:
    def __init__(self, seed=None, n_quasar=None, redshift_min=0.2, redshift_max=4.,
                 sky_radius=None, wcs=None, verbose=True, **kwargs):
        """Function that attempts to return a realistic distribution of quasar properties."""
        """
        NOTE: Currently quite arbitrary flux scaling!
        The values are chosen to be convenient for analysis, and are not realistic!

        Returns redshift, flux, ra, dec
        flux in units W/m**2
        """
        lum_solar = 3.846e26  # Solar luminosity, in Watts
        ly = 9.4607e15  # one light year, in meters
        pi = np.pi
        pixel_scale_degrees = wcs.getPixelScale().asDegrees()
        pix_origin_offset = 0.5
        x_center, y_center = wcs.getPixelOrigin()
        x_center += pix_origin_offset
        y_center += pix_origin_offset
        # NOTE: This uses the same distance parameters as the stellar distribution
        # This is of course incorrect, since quasars are much farther away
        # But, it is useful to make sure we have a range of quasar properties
        # that are measureable in an image along with stars.
        max_quasar_dist = 10000.0  # light years
        min_quasar_dist = 100.0  # Assume we're not looking at anything close
        # Luminosity relative to the sun. Also wildly inaccurate, but taken together
        # with the close distance set above, should provide measureable values in the simulated image
        min_luminosity = 0.5
        max_luminosity = 5.0
        luminosity_to_flux = lum_solar/(4.0*pi*(ly**2.0))
        rand_gen = np.random
        if seed is not None:
            rand_gen.seed(seed)

        redshift = rand_gen.uniform(redshift_min, redshift_max, size=n_quasar)
        luminosity = rand_gen.uniform(min_luminosity, max_luminosity, size=n_quasar)
        d_min = np.sqrt((rand_gen.uniform(min_quasar_dist, max_quasar_dist, size=n_quasar)**2.0 +
                         rand_gen.uniform(min_quasar_dist, max_quasar_dist, size=n_quasar)**2.0))
        distance_attenuation = (d_min + rand_gen.uniform(0, max_quasar_dist - min_quasar_dist, size=n_quasar))
        star_radial_dist = np.sqrt((rand_gen.uniform(-sky_radius, sky_radius, size=n_quasar)**2.0 +
                                    rand_gen.uniform(-sky_radius, sky_radius, size=n_quasar)**2.0)/2.0)
        star_angle = rand_gen.uniform(0.0, 2.0*pi, size=n_quasar)
        pseudo_x = x_center + star_radial_dist*np.cos(star_angle)/pixel_scale_degrees
        pseudo_y = y_center + star_radial_dist*np.sin(star_angle)/pixel_scale_degrees

        ra = []
        dec = []
        for _i in range(n_quasar):
            ra_star, dec_star = wcs.pixelToSky(geom.Point2D(pseudo_x[_i], pseudo_y[_i]))
            ra.append(ra_star)
            dec.append(dec_star)
        flux = luminosity*luminosity_to_flux/distance_attenuation**2.0
        self.raw_flux = flux
        self.redshift = redshift
        self.ra = ra
        self.dec = dec
