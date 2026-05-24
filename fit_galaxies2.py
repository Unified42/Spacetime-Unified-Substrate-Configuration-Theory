import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import trapezoid

c_km_s = 299792.458

# ============================================================
# 1. 读取 Pantheon+ 数据
# ============================================================
data_file = r"C:\Users\Administrator\Desktop\PantheonPlusSH0ES.dat"
df = pd.read_csv(data_file, sep=r'\s+', comment='#', engine='python')
z_all = df['zHD'].values
mu_obs_all = df['MU_SH0ES'].values
mu_err_all = df['MU_SH0ES_ERR_DIAG'].values
is_cal = df['IS_CALIBRATOR'].values
mask = (is_cal == 0)
z = z_all[mask]
mu_obs = mu_obs_all[mask]
mu_err = mu_err_all[mask]
print(f"读取 {len(df)} 条，哈勃流 {len(z)}")

# ============================================================
# 2. 基底开关式相变模型
# ============================================================
def H_phase(z, H0, Om, zt, alpha):
    Omega_def = (1 - Om) / (1 + (z / zt)**alpha)
    return H0 * np.sqrt(Om * (1+z)**3 + Omega_def)

def mu_phase(z_arr, H0, Om, zt, alpha):
    dL = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        z_int = np.linspace(0, zi, 200)
        Hz_int = H_phase(z_int, H0, Om, zt, alpha)
        dL[i] = (1+zi) * trapezoid(c_km_s/Hz_int, z_int)
    return 5*np.log10(dL) + 25

# ============================================================
# 3. 拟合（H0固定73）
# ============================================================
def chi2(params):
    Om, zt, alpha = params
    return np.sum(((mu_obs - mu_phase(z, 73.0, Om, zt, alpha)) / mu_err)**2)

res = minimize(chi2, x0=[0.3, 0.5, 6.0], bounds=[(0.01,0.99),(0.01,5),(0.1,15)])
Om_b, zt_b, a_b = res.x
chi2_val = res.fun
dof = len(z)-3
print(f"\n========== 开关式相变模型 ==========")
print(f"固定 H0 = 73.0")
print(f"最佳: Ωm={Om_b:.4f}, zt={zt_b:.4f}, α={a_b:.2f}")
print(f"χ²/dof = {chi2_val:.1f}/{dof} = {chi2_val/dof:.2f}")

# ΛCDM基线
def mu_LCDM(z_arr, H0, Om):
    dL = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        z_int = np.linspace(0, zi, 200)
        dL[i] = (1+zi)*trapezoid(c_km_s/(H0*np.sqrt(Om*(1+z_int)**3+(1-Om))), z_int)
    return 5*np.log10(dL)+25
chi2_lcdm = np.sum(((mu_obs - mu_LCDM(z,73,0.315))/mu_err)**2)
print(f"\nΛCDM: χ²={chi2_lcdm:.1f}/{len(z)}={chi2_lcdm/len(z):.2f}")

# ============================================================
# 4. 哈勃张力预测
# ============================================================
# 定义早期H0：如果CMB实验用ΛCDM去拟合z≈1100的宇宙，而真实宇宙是基底开关模型
# 简便方法：计算z=1100时H(z)/(1+z)的有效值，并与73比较
H_cmb = H_phase(1100, 73.0, Om_b, zt_b, a_b)
H0_early_eff = H_cmb / (1+1100)
# 更严格：计算声视界处的H(z)，然后映射到等效H0。这里用近似。
print(f"\n--- 哈勃张力预测 ---")
print(f"在 z=1100, H(z)={H_cmb:.1f}")
print(f"等效早期 H0 = H(1100)/(1+z) = {H0_early_eff:.1f} km/s/Mpc")
print(f"晚期局部 H0 = 73.0")
print(f"张力 = {73.0 - H0_early_eff:.1f} km/s/Mpc")

# ============================================================
# 5. 绘图
# ============================================================
z_line = np.logspace(-3, np.log10(0.15), 100)
plt.figure(figsize=(10,5))
plt.errorbar(z, mu_obs, yerr=mu_err, fmt='.', alpha=0.3)
plt.plot(z_line, mu_phase(z_line,73,Om_b,zt_b,a_b), 'r-', label='Phase-switch')
plt.plot(z_line, mu_LCDM(z_line,73,0.315), 'b--', label='LCDM')
plt.xlabel('z'); plt.ylabel('μ'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('hubble_switch_final.png', dpi=150); plt.show()

plt.figure(figsize=(10,4))
res_p = mu_obs - mu_phase(z,73,Om_b,zt_b,a_b)
res_l = mu_obs - mu_LCDM(z,73,0.315)
plt.errorbar(z, res_p, yerr=mu_err, fmt='.', alpha=0.3, color='red')
plt.errorbar(z, res_l, yerr=mu_err, fmt='.', alpha=0.3, color='blue')
plt.axhline(0, color='k', ls='--')
plt.xlabel('z'); plt.ylabel('Residual'); plt.legend(['Phase','LCDM'])
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig('residuals_switch_final.png', dpi=150); plt.show()