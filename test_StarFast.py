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

from astropy import units
import numpy as np
from numpy.fft import irfft2, fftshift
from scipy import constants
from lsst.afw.coord import Observatory, Weather
import lsst.afw.geom as afwGeom
from lsst.afw.geom import degrees
import lsst.afw.table as afwTable
from lsst.sims.photUtils import PhotometricParameters
import unittest
import lsst.utils.tests
from StarFast import (_wavelength_iterator, _CoordsXY, _load_bandpass, _dcr_generator,
                      _star_gen, _StellarDistribution, _create_wcs, _sky_noise_gen)

lsst_lat = -30.244639*degrees
lsst_lon = -70.749417*degrees
lsst_alt = 2663.
lsst_temperature = 20.*units.Celsius  # in degrees Celcius
lsst_humidity = 40.  # in percent
lsst_pressure = 73892.*units.pascal

lsst_weather = Weather(lsst_temperature.value, lsst_pressure.value, lsst_humidity)
lsst_observatory = Observatory(lsst_lon, lsst_lat, lsst_alt)


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
        self.wavelen, self.sb = self.getBandpass()

    def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
        """Mimic the calc_eff_wavelen method of the real bandpass class."""
        if wavelength_min is None:
            wavelength_min = self.wavelen_min
        if wavelength_max is None:
            wavelength_max = self.wavelen_max
        return((wavelength_min + wavelength_max)/2.0)

    def calc_bandwidth(self):
        f0 = constants.speed_of_light/(self.wavelen_min*1.0e-9)
        f1 = constants.speed_of_light/(self.wavelen_max*1.0e-9)
        f_cen = constants.speed_of_light/(self.calc_eff_wavelen()*1.0e-9)
        return(f_cen*2.0*(f0 - f1)/(f0 + f1))

    def getBandpass(self):
        """Mimic the getBandpass method of the real bandpass class."""
        wl_gen = _wavelength_iterator(self)
        wavelengths = [wl[0] for wl in wl_gen]
        wavelengths += [self.wavelen_max]
        bp_vals = [1]*len(wavelengths)
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

    def calcADU(self, bandpass, photParams):

        photon_energy = constants.Planck*constants.speed_of_light/(bandpass.calc_eff_wavelen()/1e9)
        photons_per_jansky = (1e-26*(photParams.effarea/1e4) *
                              bandpass.calc_bandwidth()/photon_energy)

        counts_per_jansky = photons_per_jansky/photParams.gain
        bandpass_vals = np.interp(self.wavelen, bandpass.wavelen, bandpass.sb, 0., 0.)
        simple_adu = np.sum(self.flambda*bandpass_vals)*counts_per_jansky
        return(simple_adu)


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
        self.flag_array = np.array([False]*self.n_star)
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
        self.assertAlmostEqual(2*self.pad_image*self.x_size, self.coords.xsize())
        self.assertAlmostEqual(2*self.pad_image*self.y_size, self.coords.ysize())

    def test_coord_size_over_scale_nonint(self):
        """Oversampling must only by integer factors."""
        self.coords.set_oversample(2.3)
        self.assertAlmostEqual(2*self.pad_image*self.x_size, self.coords.xsize())
        self.assertAlmostEqual(2*self.pad_image*self.y_size, self.coords.ysize())

    def test_coord_pixel_scale_base(self):
        """Make sure everything gets set, and the math is correct."""
        self.assertEqual(self.pixel_scale, self.coords.scale())

    def test_coord_pixel_scale_over(self):
        """Make sure everything gets set, and the math is correct."""
        self.coords.set_oversample(2)
        self.assertEqual(self.pixel_scale/2, self.coords.scale())

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
        bright_x = 2*self.x_loc[self.flag_array == bright_condition]
        bright_y = 2*self.y_loc[self.flag_array == bright_condition]
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
        self.weather = lsst_weather
        self.observatory = lsst_observatory
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
        dcr_gen = _dcr_generator(bp, weather=self.weather, observatory=self.observatory,
                                 pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min)/bp.wavelen_step))
        for _i in range(n_step):
            self.assertAlmostEqual(next(dcr_gen), zenith_dcr)
        with self.assertRaises(StopIteration):
            next(dcr_gen)

    def test_dcr_values(self):
        """Check DCR against pre-computed values."""
        azimuth = 0.0
        elevation = 50.0
        dcr_vals = [1.30271919792, 1.09190646505, 0.873994068277, 0.671023761839, 0.482024401891,
                    0.306093035352, 0.141471646253, -0.0124702839868, -0.156447638044, -0.291614446648,
                    -0.418830629658, -0.538108683346, -0.65051491904, -0.749927550502, -0.80762260132]
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, weather=self.weather, observatory=self.observatory,
                                 pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min)/bp.wavelen_step))
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
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min)/bp.wavelen_step))
        self.assertEqual(n_step + 1, len(bandpass_vals))


class StarGenTestCase(lsst.utils.tests.TestCase):
    """Test the flux calculation for a single star."""

    def setUp(self):
        """Define parameters used by every test."""
        self.bandpass = _BasicBandpass(band_name='g', wavelength_step=10)
        self.bandpass_highres = _BasicBandpass(band_name='g', wavelength_step=1)
        self.photParams = PhotometricParameters(exptime=30.0, nexp=1, platescale=.7, bandpass='g')
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
        star_gen = _star_gen(source_record=self.source_rec, bandpass=self.bandpass, verbose=False,
                             bandpass_highres=self.bandpass_highres, photParams=self.photParams)
        spectrum = np.array([flux for flux in star_gen])
        pre_comp_spectrum = np.array([5.7637979669137405, 5.93354511786311, 6.083468704742461,
                                      6.21396966086581, 6.325613048713335, 6.4190942766733645,
                                      6.495208931778466, 6.554826236433618, 6.598866014715503,
                                      6.628278970967843, 6.644030031500907, 6.647084471932508,
                                      6.638396542012121, 6.618900301765851, 4.616292142671812])
        abs_diff_spectrum = np.sum(np.abs(spectrum - pre_comp_spectrum*65315651.))
        self.assertAlmostEqual(abs_diff_spectrum, 0.0)

    def test_sed_spectrum(self):
        """Check a spectrum defined by an SED against pre-computed values."""
        sed_list = [_BasicSED(self.source_rec["temperature"])]
        star_gen = _star_gen(sed_list=sed_list, source_record=self.source_rec,
                             bandpass_highres=self.bandpass_highres,
                             bandpass=self.bandpass, verbose=True, photParams=self.photParams)
        spectrum = np.array([flux for flux in star_gen])
        pre_comp_spectrum = np.array([0.1337338, 0.13699958, 0.14026536, 0.14353114,
                                      0.14679693, 0.15006271, 0.15332849, 0.15659427,
                                      0.15986006, 0.16312584, 0.16639162, 0.1696574,
                                      0.17292319, 0.17618897, 0.12527542])
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
        pixel_radius = np.sqrt(((self.x_size/2.0)**2.0 + (self.y_size/2.0)**2.0)/2.0)
        self.sky_radius = pixel_radius*self.wcs.getPixelScale().asDegrees()

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
            _x, _y = self.wcs.skyToPixel(afwGeom.SpherePoint(_ra, _dec))
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
        dimension = np.sqrt(CoordsXY.xsize()*CoordsXY.ysize())
        self.assertLess(np.abs(np.std(noise_image) - self.amplitude), 1.0/dimension)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Temp."""

    pass


def setup_module(module):
    """Temp."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
