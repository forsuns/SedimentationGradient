import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import rcParams
from scipy.io import loadmat
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FormatStrFormatter

# 전체 글꼴 설정
def set_global_styles():
    rcParams['font.family'] = 'Arial'  # Arial 등 다른 폰트로 설정 가능
    rcParams['font.size'] = 16

# 데이터 로드 함수
def load_data():
    avg_data = pd.read_csv('./avg41.dat', delim_whitespace=True, header=None, names=["ll", "xxx", "yyy", "bot", "spd"])
    sed1_data = pd.read_csv('../../Run/ss41/sed1.dat', delim_whitespace=True, skiprows=1, header=None,
                            names=["ll", "ii", "jj", "xxx", "yyy", "bot1", "belv", "elv"])
    sed31_data = pd.read_csv('../../Run/ss41/sed31.dat', delim_whitespace=True, skiprows=1, header=None,
                             names=["ll", "ii", "jj", "xxx", "yyy", "bot2", "belv", "elv"])
    basemap_data = loadmat('../../Pre/basemap/o02Depth/basemap.mat')
    island_data = pd.read_csv('../../Pre/basemap/islandNames.txt', delim_whitespace=True, header=None,
                              names=["inX", "inY", "inames"])
    return avg_data, sed1_data, sed31_data, basemap_data, island_data

# Sedimentation 계산 함수
def calculate_sedimentation(avg_data, sed1_data, sed31_data):
    alpha, rho0, Cx, vsink = 1, 1025.0, 1.0, 0.0000036
    h = -avg_data["bot"]
    spd = avg_data["spd"]
    sedg = (alpha / spd) * (Cx / rho0) * (vsink / h)
    sedg[sedg < 0] = 0  # 음수 값 제거
    vel1 = (sed31_data["bot2"] - sed1_data["bot1"]) * 1000
    sedg[h <= 4] = np.nan
    sedg[spd <= 0.00001] = np.nan
    return sedg, vel1

# 등고선 플롯 생성 함수
def plot_contour(ax, avg_data, sedg, xmin, xmax, ymin, ymax, flucMin, flucMax, cStep, csort):
    gx, gy = np.meshgrid(np.arange(xmin, xmax, csort), np.arange(ymin, ymax, csort))
    zi = griddata((avg_data["xxx"], avg_data["yyy"]), sedg, (gx, gy), method='linear')
    zi = np.clip(zi, flucMin, flucMax)
    levels = np.arange(flucMin, flucMax, cStep)
    contour = ax.contourf(gx, gy, zi, levels=levels, cmap='jet', extend='both')
    return contour

# 컬러바 설정 함수
def configure_colorbar(ax, contour, fig):
    cbar = fig.colorbar(contour, ax=ax, location='right', fraction=0.05, pad=0.02)
    cbar.ax.set_aspect(30)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    cbar.extend = False
    cbar.set_label('kg/m')
    cbar.outline.set_linewidth(0)

# 지도와 섬 이름 추가 함수
def add_map_features(ax, basemap_data, island_data, xmin, xmax, ymin, ymax):
    xx, yy = basemap_data['xx'], basemap_data['yy']
    ax.fill(xx, yy, color=[1, 0.88, 0.21], edgecolor='black', linewidth=0.5)
    for _, row in island_data.iterrows():
        if xmin <= row['inX'] <= xmax and ymin <= row['inY'] <= ymax:
            ax.text(row['inX'], row['inY'], row['inames'], fontsize=12, color='black', ha='center', va='center')

# 북쪽 방향 표시 함수
def add_north_arrow(ax, x=0.9, y=0.9, width=0.02, height=0.1, label="N", fontsize=12, color='black', edgecolor='black'):
    shadow_effect = withStroke(linewidth=3, foreground="gray", alpha=0.6)
    ax.annotate('', xy=(x, y), xytext=(x, y - height),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor=color, edgecolor=edgecolor, linewidth=0.5, width=width * 100,
                                headwidth=width * 300, headlength=height * 200))
    ax.text(x, y + height * 0.2, label, fontsize=fontsize, color=color,
            ha='center', va='center', transform=ax.transAxes, path_effects=[shadow_effect])

# 전체 실행 함수
def main():
    set_global_styles()
    xmin, xmax, ymin, ymax = 385000, 425000, 3850000, 3886000
    flucMin, flucMax, cStep = 0.0, 8e-9, 1e-10
    csort = 10

    # 데이터 로드
    avg_data, sed1_data, sed31_data, basemap_data, island_data = load_data()

    # Sedimentation 계산
    sedg, vel1 = calculate_sedimentation(avg_data, sed1_data, sed31_data)

    # 플롯 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = plot_contour(ax, avg_data, sedg, xmin, xmax, ymin, ymax, flucMin, flucMax, cStep, csort)
    configure_colorbar(ax, contour, fig)
    add_map_features(ax, basemap_data, island_data, xmin, xmax, ymin, ymax)
    add_north_arrow(ax, x=0.95, y=0.95, width=0.04, height=0.05, fontsize=14)
    plt.title('Sedimentation Gradient', fontsize=16)

    # X, Y 축 설정 및 저장
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('UTM Coordinate, Easting (m)', fontsize=16)
    ax.set_ylabel('Northing (m)', fontsize=16)
    plt.tight_layout()
    #plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)  # 플롯 영역 고정
    plt.savefig('ss41_sdr.png', dpi=200, bbox_inches='tight')
    plt.show()

# 실행
if __name__ == "__main__":
    main()