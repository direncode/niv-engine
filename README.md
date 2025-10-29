<div align="center">

# National Impact Velocity (NIV) Engine  
### *Computational Implementation of Capital Velocity Economics (CVE)*  

**Author:** Diren Kumaratilleke  
**Affiliation:** Undergrad at University of North Carolina Chapel Hill  

---

<p align="center">
  <em>
  “An economy is not measured by how much it produces,  
  but by how fast its capital regenerates and flows through itself.”  
  </em>
</p>

---

### Core Equation

\[
NIV_t = \frac{u_t \cdot P_t^2}{(X_t + F_t)^\eta}
\]

| Symbol | Meaning |
|:--:|:--|
| \(u_t\) | Activation intensity (derived from investment, liquidity, and rate shifts) |
| \(P_t\) | Regeneration share of output (R&D + Education + Capital Formation) |
| \(X_t\) | Idle capacity (underutilized production share) |
| \(F_t\) | Friction (financial spread and debt drag) |
| \(\eta\) | Friction exponent |

</div>

---

## Overview
The **National Impact Velocity (NIV)** framework models the *real-time economic metabolism* of a nation through  
**capital throughput**, **regeneration**, and **frictional drag**.  
It replaces static GDP with a *motion-based metric* that captures the intensity, efficiency, and regenerative power of national production.

NIV is the computational implementation of **Capital Throughput Economics (CVE)** — a new paradigm uniting fiscal, monetary, and real-sector data under a single regenerative measure.

---

## System Architecture

| Directory | Key Files | Description |
|------------|------------|-------------|
| **src/** | `fred_fetch.py`, `fred_live.py`, `niv_visualization_v6.py`, `config.yaml` | Core engine scripts and configuration |
| **docs/** | `NIV_Computational_Maths.pdf`, `NIV_Code_Formula_Mapping.pdf` | Documentation and methodology papers |
| **outputs/** | `visuals/v6/`, `niv_processed_v6_*.csv` | Generated datasets and visualizations |

---

## ⚙️ Configuration

All parameters are controlled through `src/config.yaml`:
```yaml
config:
  mode: empirical
  steps: 240
  scenario: base_case

params:
  eta: 1.25
  alpha1: 0.8
  alpha2: 0.6
  alpha3: -0.3
  beta1: 0.4
  beta2: 0.4
  beta3: 0.2
  lambda: 0.05

simulation:
  g_growth: 0.02
  m2_growth: 0.03
  rate_change: -0.01
  output_dir: visuals/v6

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run NIV Engine
python src/niv_visualization_v6.py --config src/config.yaml




