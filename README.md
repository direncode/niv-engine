# National Impact Velocity (NIV) Engine v6

The **National Impact Velocity (NIV)** framework models the real-time economic metabolism of a nation through capital throughput, regeneration, and frictional drag.  
It operationalizes the core equation:

\[
NIV_t = \frac{u_t \cdot P_t^2}{(X_t + F_t)^\eta}
\]

where:
- \( u_t \): Activation intensity (derived from investment, liquidity, and rate shifts)  
- \( P_t \): Regeneration share of output (R&D + Education + Capital Formation)  
- \( X_t \): Idle capacity (underutilized production share)  
- \( F_t \): Friction (financial spread and debt drag)  
- \( \eta \): Friction exponent  

---

## 🌐 System Architecture
src/
├── fred_fetch.py # Historical FRED data downloader
├── fred_live.py # Live FRED API fetcher
├── niv_make_gdpnorm_v3.py # Core NIV normalization logic
├── niv_visualization_v6.py # Main computational engine + visualizations
├── check_data.py # Sanity and consistency checks
└── config.yaml # Scenario configuration file

docs/
├── NIV_Computational_Maths.pdf
└── NIV_Code_Formula_Mapping.pdf

outputs/
├── visuals/v6/ # Generated plots and CSVs
└── *.csv # Simulation and processed data


---

## ⚙️ Configuration

Edit the `config.yaml` file to modify parameters, such as:
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

--- 

##Running the Engine##

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run NIV Engine
python src/niv_visualization_v6.py --config src/config.yaml

Results are saved under visuals/v6/:

niv_gdpnorm – National Impact Velocity trajectory

niv_impulse – Throughput impulse (ΔNIV)

niv_drag – Structural drag (rolling σ of NIV)

niv_lsi – Liquidity stress intensity