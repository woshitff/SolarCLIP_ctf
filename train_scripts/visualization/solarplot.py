import numpy as np
import matplotlib.pyplot as plt

import sunpy.map
from astropy.io import fits

INSTRUME_DICT = {
    '0094': 'AIA_4',
    '0131': 'AIA_1',
    '0171': 'AIA_3',
    '0193': 'AIA_2',
    '0211': 'AIA_2',
    '0304': 'AIA_4',
    '0335': 'AIA_1',
    '1600': 'AIA_3',
    '1700': 'AIA_3',
    '4500': 'AIA_3'
}

WAVELNTH_DICT = {
    '0094': 94,
    '0131': 131,
    '0171': 171,
    '0193': 193,
    '0211': 211,
    '0304': 304,
    '0335': 335,
    '1600': 1600,
    '1700': 1700,
    '4500': 4500
}


def get_header(modal: str,
               time: str):
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = 1024
    header['NAXIS2'] = 1024
    header['IMG_TYPE'] = 'LIGHT'

    header['TELESCOP'] = 'SDO/AIA'
    header['DATE-OBS'] =  time       #'2010-06-03T00:00:08.14'
    header['INSTRUME'] = INSTRUME_DICT.get(modal)
    header['WAVELNTH'] = WAVELNTH_DICT.get(modal)
    
    header['WAVEUNIT'] = 'angstrom'

    header['CTYPE1'] = 'HPLN-TAN'
    header['CUNIT1'] = 'arcsec'
    header['CRVAL1'] = 0.0
    header['CDELT1'] = 2.4
    header['CRPIX1'] = 512.5
    header['CTYPE2'] = 'HPLT-TAN'
    header['CUNIT2'] = 'arcsec'
    header['CRVAL2'] = 0.0
    header['CDELT2'] = 2.4
    header['CRPIX2'] = 512.5
    header['CROTA2'] = 0.0

    return header

def solarplot(data: np.array,
              modal: str,
              time: str,
              save_path: str
              ):
    header = get_header(modal, time)
    mymap = sunpy.map.Map((data, header))

    mymap.plot()
    # plt.colorbar()
    plt.savefig(save_path)

    
                