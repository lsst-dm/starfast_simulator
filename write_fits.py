#
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

"""Persist python arrays as fits files readable as Eimages."""

import numpy as np
import pyfits

lsst_lat = -30.244639
lsst_lon = -70.749417
lsst_ccd_xsize = 4000
lsst_ccd_ysize = 4072


class PersistArray:
    """Generate metadata and write the fits file to disk."""

    def __init__(self, array, ra=lsst_lat, dec=lsst_lon, pixel_scale=None, sky_rotation=0.0, zenith_angle=0.0,
                 azimuth_angle=0.0, filter_name='g', seed=None, obsid=None):
        """Set up metadata."""
        self.ra = ra
        self.dec = dec
        self.epoch = 2000.0
        hour_angle = zenith_angle * np.cos(np.radians(azimuth_angle)) / 15.0
        self.mjd = 59000.0 + (lsst_lat / 15.0 - hour_angle) / 24.0
        self.filter = filter_name
        self.scale = pixel_scale / 3600.0
        self.rot_angle = sky_rotation
        self.seed = seed
        self.zenith = zenith_angle
        self.azimuth = azimuth_angle
        self.obsid = obsid
        self.array = array
        self.y_size, self.x_size = array.shape

        self._make_header()
        self.filename = "lsst_e_%i_f2_R22_S11_E000.fits" % obsid

    def _make_header(self):
        fitsHeader = pyfits.Header()
        fitsHeader.set("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        fitsHeader.set("OUTFILE", "SnapId_000")
        fitsHeader.set("RADESYS", "ICRS")
        fitsHeader.set("EQUINOX", self.epoch)
        fitsHeader.set("CRVAL1", self.ra)
        fitsHeader.set("CRVAL2", self.dec)
        fitsHeader.set("CRPIX1", self.x_size // 2 + 1)  # the +1 is because LSST uses 0-indexed images
        fitsHeader.set("CRPIX2", self.y_size // 2 + 1)  # FITS files use 1-indexed images
        fitsHeader.set("CTYPE1", "RA---TAN")
        fitsHeader.set("CTYPE2", "DEC--TAN")
        fitsHeader.set("CD1_1", self.scale * np.cos(np.radians(self.rot_angle)))
        fitsHeader.set("CD1_2", -self.scale * np.sin(np.radians(self.rot_angle)))
        fitsHeader.set("CD2_1", self.scale * np.sin(np.radians(self.rot_angle)))
        fitsHeader.set("CD2_2", self.scale * np.cos(np.radians(self.rot_angle)))

        fitsHeader.set("TAI", self.mjd)
        fitsHeader.set("MJD-OBS", self.mjd)

        fitsHeader.set("EXTTYPE", "IMAGE")
        fitsHeader.set("EXPTIME", 30.0)
        fitsHeader.set("AIRMASS", 1.0 / np.cos(np.radians(self.zenith)))
        fitsHeader.set("ZENITH", self.zenith)
        fitsHeader.set("AZIMUTH", self.azimuth)
        fitsHeader.set("FILTER", self.filter)
        if self.seed is not None:
            fitsHeader.set("SEED", self.seed)
        if self.obsid is not None:
            fitsHeader.set("OBSID", self.obsid)

        self.fitsHeader = fitsHeader

    def write(self, directory, clobber=True, add_noise=None, **kwargs):
        """Write the fits file."""
        # array = self._fill_image(**kwargs)
        array = self.array
        if add_noise is not None:
            array += np.abs(np.random.normal(scale=add_noise, size=array.shape))
        hdu = pyfits.PrimaryHDU(array, header=self.fitsHeader)
        hdu.writeto(directory + self.filename, clobber=clobber)

    def _fill_image(self, x0=None, y0=None):
        y_size_img, x_size_img = self.array.shape
        if x0 is None:
            x0 = (lsst_ccd_xsize - self.x_size) // 2
        if y0 is None:
            y0 = (lsst_ccd_ysize - self.y_size) // 2
        x1 = int(x0 + x_size_img)
        y1 = int(y0 + y_size_img)
        array_fill = np.zeros((self.y_size, self.x_size))
        array_fill[y0:y1, x0:x1] = self.array
        self.array = array_fill
        
        return(array_fill)
