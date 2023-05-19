import matplotlib
import matplotlib.pyplot as plt

CM = 1/2.54
DPI = 500
# DPI = 100
FONT = 'Arial'
FONT_SIZE = 7
# TEXTWIDTH = 18.35  # elsevier 5p
# COLUMNWIDTH = 9.0

font = {'family': FONT,
        'sans-serif': FONT,
        # 'weight': 'bold',
        'size': FONT_SIZE}
matplotlib.rc('font', **font)
# print(matplotlib.get_cachedir())

plt.rcParams.update({
        'figure.raise_window': False,
        'figure.dpi': DPI,
})

plt.rcParams.update({
        'ps.fonttype': 42,
        'pdf.fonttype': 42,
        'image.interpolation': 'none'
})

from matplotlib import font_manager
plt.set_loglevel('info')
fm = matplotlib.font_manager
# fm._call_fc_list.cache_clear()
fm._get_fontconfig_fonts.cache_clear()
fs = sorted(fm._get_fontconfig_fonts())

fm.findfont(FONT, rebuild_if_missing=True)
# print(font_manager.fontManager.ttflist)

