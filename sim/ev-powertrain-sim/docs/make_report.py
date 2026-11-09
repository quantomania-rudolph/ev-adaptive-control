"""Builds the in-depth explainer PDF: python docs/make_report.py  ->  docs/Powertrain_Sim_Explainer.pdf"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether, Preformatted, HRFlowable)

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf")); pdfmetrics.registerFont(TTFont("DV-M", FD + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

INK, MUTED, ACC, ORG = colors.HexColor("#14181f"), colors.HexColor("#5a616b"), colors.HexColor("#1c5cab"), colors.HexColor("#c2410c")
P = lambda n, **k: ParagraphStyle(n, fontName=k.pop("f", "DV"), textColor=k.pop("c", INK), **k)
title = P("t", f="DV-B", fontSize=22, leading=26, spaceAfter=4)
sub = P("s", fontSize=10, leading=14, c=MUTED)
h1 = P("h1", f="DV-B", fontSize=14, leading=18, c=ACC, spaceBefore=14, spaceAfter=5, keepWithNext=1)
h2 = P("h2", f="DV-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, keepWithNext=1)
body = P("b", fontSize=9, leading=13.2, spaceAfter=5)
bl = P("bl", fontSize=9, leading=13, leftIndent=13, bulletIndent=2, spaceAfter=2.5)
code = P("c", f="DV-M", fontSize=8, leading=11, leftIndent=8, backColor=colors.HexColor("#f1f3f6"), borderPadding=(4, 4, 4, 4), spaceBefore=3, spaceAfter=8)
cap = P("cap", f="DV-I", fontSize=8, leading=11, c=MUTED, spaceAfter=8)
cell = P("cell", fontSize=8, leading=10.5)
cellb = P("cellb", f="DV-B", fontSize=8, leading=10.5)
note = P("note", fontSize=8.8, leading=12.6, backColor=colors.HexColor("#fdf3ec"), borderPadding=(6, 6, 6, 6), borderColor=ORG, borderWidth=0.6, spaceBefore=6, spaceAfter=10)

S = []
def H1(t): S.append(Paragraph(t, h1))
def H2(t): S.append(Paragraph(t, h2))
def B(t): S.append(Paragraph(t, body))
def L(items): [S.append(Paragraph(i, bl, bulletText="•")) for i in items]
def C(t): S.append(Preformatted(t.strip("\n"), code))
def N(t): S.append(Paragraph(t, note))
def FIG(path, w, caption):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    S.append(KeepTogether([Image(path, width=w, height=w * ih / iw), Paragraph(caption, cap)]))
def T(rows, widths, head=True, zebra=True):
    data = [[Paragraph(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5d9df")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf4"))]
    t.setStyle(TableStyle(st)); S.append(t); S.append(Spacer(1, 7))

# ======================================================================================================== COVER
S += [Paragraph("EV Powertrain Simulation: what we built and why", title),
      Paragraph("Explainer for review. Adaptive Control under Input/State Constraints (BIBS stability under sudden actuator saturation). "
                "Code, data and tests are in the project folder <font face='DV-M'>ev-powertrain-sim</font>. Written October 27, 2026.", sub),
      HRFlowable(width="100%", thickness=0.8, color=INK, spaceBefore=6, spaceAfter=8)]

H1("0. How to use this document")
B("This is written so a reader who has not seen the code can follow every part, check the physics, and challenge the choices. "
  "Each section says <b>what the part does</b>, <b>why it was built that way</b>, <b>what was verified and how</b>, and <b>what is assumed or still wrong</b>. "
  "Section 11 lists questions worth putting to a reviewer. Section 10 lists the mistakes we found in our own work, because those are the most useful evidence of how far to trust the rest.")
N("<b>Reading rule.</b> Anything marked <b>Verified</b> was measured or tested in the sandbox, and the number is quoted. Anything marked <b>Assumed</b> is a judgement, with no evidence behind it yet. "
  "Nothing in this report claims a stability proof. The Lyapunov decay bound is still open.")

H1("1. The project in plain words")
B("An electric car's motor controller asks the inverter for a voltage. If the battery or inverter cannot deliver it, the controller still believes it did. "
  "A controller that <i>learns</i> a motor parameter online (an adaptive controller) can then mistake the shortfall for a wrong parameter, keep changing its estimate, and run away. "
  "The research question is how to guarantee the states stay bounded when that happens. The coding team's job is a simulation faithful enough to show the failure, "
  "test the fixes, and expose, as data, every place where what was <b>requested</b> differs from what was <b>delivered</b>.")
B("The notes name the plant form <font face='DV-M'>x' = f(x) + g(x)u + d(x,t)</font> and the saturation form <font face='DV-M'>u = sat(u_c), Δu = u − u_c</font>. "
  "The whole simulation is organised so that each of those symbols is a column of data.")

H1("2. Architecture")
FIG("docs/figures/fig_architecture.png", 6.9 * inch, "Figure 1. Data flows left to right along the top; energy flows between plant and battery; telemetry watches everything.")
T([["Part", "File", "Job in one sentence"],
   ["Cycle loader", "cycles.py", "Turns any messy speed-vs-time file into a clean, checked speed request and records every repair."],
   ["Vehicle mapping", "cycles.py: VehicleEq", "Maps vehicle speed to motor speed and reflects mass, drag, rolling and grade onto the motor shaft."],
   ["Controller", "control.py", "Speed loop, adaptive current loop, anti-windup, and a governor that stops it asking for the impossible."],
   ["Plant adapter", "plant.py", "Wraps the Gym-Electric-Motor simulator; does the voltage saturation itself so it can be measured."],
   ["Battery + DC link", "battery.py", "Pack voltage, resistance, current limits and the bus capacitor; makes the voltage ceiling a state."],
   ["Telemetry", "telemetry.py", "Computes the f, g, d split, mismatch rows and energy books; holds the channel catalog."],
   ["Experiments", "run_exp.py, run_cycle.py", "Scenario runners for the 8 s experiments and for full real drives."],
   ["Viewer", "viewer/", "Self-contained page that plots every channel for any two runs."],
   ["Tests", "tests/", "60 automated checks; they pin the facts the project depends on."]],
  [1.15 * inch, 1.5 * inch, 4.25 * inch])

# ======================================================================================================== PHYSICS
H1("3. The physics we are simulating")
H2("3.1 The motor (a permanent-magnet synchronous motor, PMSM)")
B("A three-phase motor is easier to control after a change of coordinates (the dq frame) that rotates with the rotor. In that frame the motor is two coupled first-order circuits plus a mechanical equation:")
C("""
L_d di_d/dt = u_d − R i_d + ω_el L_q i_q
L_q di_q/dt = u_q − R i_q − ω_el (L_d i_d + ψ)
T_elec      = 1.5 p (ψ + (L_d − L_q) i_d) i_q
J dω/dt     = T_elec − T_friction(ω) − T_disturbance(t)        ω_el = p ω
""")
B("<b>What each symbol means.</b> <i>i_d, i_q</i> are currents along and across the magnet; <i>u_d, u_q</i> the voltages we command; <i>ψ</i> the magnet's flux linkage (0.066 Wb, the unknown the adaptive controller learns); "
  "<i>ω_el</i> the electrical speed; <i>R, L_d, L_q</i> the winding resistance and inductances (18 mΩ, 0.37 mH, 1.2 mH); <i>p = 3</i> pole pairs. "
  "The term <i>ω_el ψ</i> is the <b>back-EMF</b>: the voltage the spinning magnet generates against us. At 200 rad/s it is about 40 V.")
B("<b>Why torque is mostly <i>i_q</i>.</b> With <i>i_d = 0</i> the torque is <i>1.5 p ψ i_q</i>, so the controller regulates <i>i_q</i> to make torque and holds <i>i_d</i> near zero. "
  "<b>Verified:</b> open-loop currents from the simulator match these equations to 1e-10 A, and torque matches the formula.")
B("<b>The speed limit of this simplification.</b> The inverter can only produce a voltage of magnitude <i>V_dc/√3</i> (173 V at 300 V). At high speed the back-EMF and cross-coupling terms use up that budget, "
  "so holding <i>i_d = 0</i> becomes impossible above a <i>base speed</i> (about 195 rad/s at 240 A for this motor). Beyond it a real car injects negative <i>i_d</i> (field weakening). "
  "Tier 1 stays below that speed on purpose; field weakening is the hardest item in Tier 2.")

H2("3.2 The inverter and why saturation is a disc")
B("The controller commands <i>(u_d, u_q)</i>. The inverter makes three phase voltages from three duty cycles in [−1, 1]. The simulator <b>clips duty cycles silently, leg by leg</b>, which makes the true limit a hexagon in the dq plane "
  "(direction dependent) and gives no record that clipping happened. <b>Verified</b> by probe scripts.")
B("We therefore do the saturation ourselves, before the simulator sees the command: <font face='DV-M'>u_app = u_cmd · min(1, u_lim/|u_cmd|)</font>, a radial projection onto a disc of radius <i>u_lim = V_dc/√3</i>. "
  "The disc is the largest circle inside the hexagon, so it is isotropic, keeps the direction of the command, and is Lipschitz, which is what a proof wants. "
  "A zero-sequence offset (<font face='DV-M'>duty −= 0.5(max+min)</font>) shifts all three legs equally, leaves the dq voltage unchanged, and lets the disc reach <i>V_dc/√3</i>. "
  "Because we pass only voltages inside the disc, the simulator never clips: the hidden-clipping check reads about 1e-11 V in every run.")

# ======================================================================================================== PLANT
H1("4. plant.py: the adapter between our controllers and the simulator")
B("<b>Design rule: controllers see only this class, never the simulator.</b> That is what lets Tier 2 replace the motor with a full vehicle without touching the controller.")
L(["<b>Time-varying load.</b> The simulator's default load holds speed constant. We subclass its polynomial load (<font face='DV-M'>RoadLoad</font>) so speed is a real state and an external torque <i>d(t)</i> can be injected. Parameters: constant (rolling) <i>a</i>, viscous <i>b</i>, quadratic (drag) <i>c</i>.",
   "<b>Constraints off.</b> The simulator ends an episode when current exceeds its limit. We pass <font face='DV-M'>constraints=()</font> so a violation is logged as data, not a crash.",
   "<b>apply_voltage(u_cmd)</b> projects onto the ceiling, steps the simulator, and returns a ledger: requested and delivered voltages, <i>Δu</i>, a saturated flag, the ceiling, and a second figure <i>M3</i> (plant-reported minus ours) that must stay at zero.",
   "<b>Battery hook.</b> If a battery is supplied, the ceiling each step is <i>V_dc/√3</i> from the battery state, and the motor's electrical power is fed back to it. The simulator's own supply stays a fixed 300 V numerical scale (the duty cycle is computed against it) because we always command the voltage we want; only our ceiling changes.",
   "<b>New constructor arguments:</b> step size <i>tau</i> (1e-4 s default, 2e-4 s for cycle runs) and the load coefficients."])
N("<b>Known limitation.</b> When regeneration pushes the real bus above 300 V, the true ceiling would be higher, but the simulator's 300 V scale would clip. We cap the ceiling at 173.2 V to keep hidden clipping at zero, which wastes up to about 4 V of headroom during regen. This bug was found by our own check (section 10).")

# ======================================================================================================== CONTROL
H1("5. control.py: the controller, and why each piece exists")
B("<b>Structure.</b> A cascade, as in a real drive: a slow speed loop produces a current request; fast current loops produce voltage requests.")
C("""
speed PI (every 10 steps) ──> i_q* (clipped to i_lim = 250 A, or to the governor limit)
q-axis ADAPTIVE current loop ──> u_q,c            d-axis certainty-equivalence loop ──> u_d,c
""")
H2("5.1 The adaptive q-axis loop (the heart of the project)")
B("The unknown is the flux linkage <i>ψ</i>. Define the tracking error <i>e = i_q − i_q*</i> and the estimate error <i>ψ~ = ψ^ − ψ</i>. The control law cancels what it knows and uses the estimate for the rest:")
C("""
u_q,c = R i_q + ω_el (L_d i_d + ψ^) − k e
Error dynamics (no saturation):   L_q de/dt = −k e + ω_el ψ~
Lyapunov function:                V = ½ L_q e² + ψ~² / (2Γ)
V-dot = −k e² + ψ~ (ω_el e + dψ^/dt / Γ)   ⇒   choose   dψ^/dt = −Γ ω_el e     (V-dot = −k e² ≤ 0)
""")
B("<b>Why the sign matters.</b> An earlier draft had the sign of the <i>ω_el ψ~</i> term wrong and the estimate diverged to −6.8 Wb even without saturation. "
  "This is the single most likely place for a future re-derivation to go wrong, so the code carries a warning. <b>Verified</b> by simulation: with no saturation the estimate converges and stays within 0.0198 Wb of truth.")
B("<b>Why the unknown multiplies speed.</b> The regressor is <i>ω_el</i>, so the estimator learns fastest exactly where the voltage ceiling is tightest (high speed).")
H2("5.2 How saturation breaks it, and the three fixes")
B("When the actuator saturates, the true error dynamics gain a term: <font face='DV-M'>L_q de/dt = −k e + ω_el ψ~ + Δu_q</font>. The adaptive law cannot tell <i>Δu_q</i> from <i>ψ~</i>, so it moves the estimate to explain an error that has nothing to do with <i>ψ</i>. That is adaptive windup.")
T([["Mode", "Idea", "What it changes"],
   ["naive (B)", "no protection", "dψ^/dt = −Γ ω_el e. The estimate is fooled by Δu_q."],
   ["robust (C)", "σ-modification + projection", "Adds a leak toward the nominal value, −σ(ψ^ − ψ_nom), and clips ψ^ to [0.03, 0.10] Wb. We leak toward nominal, not toward zero, because ψ is far from zero (the notes write the zero version; Theory should confirm the change)."],
   ["aw (D)", "anti-windup auxiliary state", "L_q dz/dt = −k z + Δu_q, and the adaptive law uses e_a = e − z. For this scalar loop z is the part of the error caused by the actuator, so e_a obeys the unsaturated error dynamics exactly and the estimate is not fooled."]],
  [0.95 * inch, 1.6 * inch, 4.35 * inch])
H2("5.3 The governor (reference governor in miniature)")
B("Even with a protected estimator, asking for current the voltage cannot sustain leaves the drive in a trap (we measured 230 A producing only 18 N·m, because the d-axis cross-coupling voltage consumed the budget). "
  "The steady-state voltage ellipse gives the largest <i>i_q</i> the ceiling can sustain at a speed: <font face='DV-M'>(L_q i_q)² + ψ² ≤ (u_max/ω_el)²</font>. The governor clips the current request to a safety factor (0.75) of that. "
  "With <font face='DV-M'>vdc_aware=True</font> it uses the <i>measured</i> bus voltage instead of the nominal one. That single change is why exposing <i>V_dc</i> as a channel matters.")

# ======================================================================================================== BATTERY
H1("6. battery.py: the battery pack and DC link")
B("<b>Why it exists.</b> The notes list DC-link voltage and battery state of charge among the states. More importantly, the saturation event is far more convincing when the powertrain produces it than when we impose a number.")
C("""
battery  (OCV(SOC), R0(T), one RC branch, BMS current limits)
   │  busbar resistance Rc
   ▼
DC-link capacitor C (voltage state V_dc)  ───►  inverter  ───►  motor
""")
L(["<b>Pack.</b> 80 cells in series, 60 Ah. Open-circuit voltage from a generic NMC curve (302.4 V at 50 %). Ohmic resistance 0.08 Ω at 25 °C rising when cold. One RC branch (0.048 Ω, 20 s) for diffusion. Heating and cooling are modelled, but temperature barely moves in 8 s.",
   "<b>Bus.</b> <font face='DV-M'>C dV_dc/dt = i_batt − i_dc</font> with <i>i_dc = P_dc / V_dc</i> and <i>P_dc = P_el + inverter loss</i> (idle 30 W plus a conduction term). <i>C = 2 mF</i>.",
   "<b>BMS limit.</b> Pack current is clamped to +250 A / −150 A. When the limit binds, the capacitor supplies the shortfall, and the bus falls in milliseconds. The voltage the limiter drops is dissipated there (<i>P_bms_clip</i>), which keeps the energy books closed.",
   "<b>Numerics.</b> The capacitor-resistor loop has a time constant of about 0.3 ms, comparable to the 0.1 ms step, so explicit Euler would be marginal. The bus update is solved implicitly (one linear equation) and the RC branch with its exact exponential."])
B("<b>Verified.</b> Resting bus equals the open-circuit voltage; a 10 kW draw gives the expected resistive drop; regeneration charges the pack; the pack's energy chain (chemical power − losses − bus power − limiter heat) closes to 1e-11 W at every step; the bus current balance closes to 0.07 W rms.")
N("<b>Assumed, not fitted.</b> Every pack number is a generic lithium-ion value. In section 8 a real Tesla trace suggests the effective pack resistance is about 0.062 Ω against our 0.085 Ω, which is a calibration task, not a verified fact.")

# ======================================================================================================== TELEMETRY
H1("7. telemetry.py: the data product")
B("The plant logs what it measures. <font face='DV-M'>derive()</font> adds everything an analyst needs, as pure functions of logged columns, so the same code applies to any run or to a future vehicle plant. "
  "<font face='DV-M'>CATALOG</font> lists 98 channels with unit, group, role and meaning.")
H2("7.1 The f, g, d split")
C("""
di_d/dt = (−R i_d + ω_el L_q i_q)/L_d  +  u_d,cmd/L_d  +  Δu_d/L_d       f_id,  g_id u,  d_id
di_q/dt = (−R i_q − ω_el(L_d i_d+ψ))/L_q + u_q,cmd/L_q + Δu_q/L_q       f_iq,  g_iq u,  d_iq
dω/dt   = (T_elec − T_fric)/J                − T_dist/J                    f_w,           d_w
""")
B("The actuator disturbance <i>d_id, d_iq</i> is literally <i>g · Δu</i>: the notes' <i>d(u,x)</i>, a disturbance that depends on the input. The road load <i>d_w</i> is the external disturbance. "
  "<b>Verified:</b> adding these terms reproduces the measured derivatives within 0.03 to 0.85 % rms (varies by run), which is the check that the columns mean what they say.")
H2("7.2 The mismatch ledger")
B("Sign convention everywhere: <b>delta = delivered − requested</b>. Negative means the stage gave less than it was asked.")
T([["Row", "Between", "Channel(s)", "Meaning"],
   ["M1", "driver → controller", "e_speed", "Speed request minus actual speed."],
   ["M2", "controller → inverter", "d_u_d, d_u_q, d_u_norm, saturated", "Voltage delivered minus requested, and a flag."],
   ["M3", "our saturation → simulator", "d_u_plant_d/q", "Must be zero: proves no hidden clipping."],
   ["M4", "speed loop → current", "i_ref_clip", "Current request after clipping minus before."],
   ["M5", "current → shaft", "d_T", "Actual torque minus the torque the current request implied."],
   ["M6", "electrical power", "dP_elec, dP_mech", "Delivered minus requested power. Use the shaft version (see caution)."],
   ["M8", "energy", "E_mech_unserved", "Cumulative shaft energy requested and not delivered."],
   ["M9", "battery → bus", "bus_sag, ceiling_loss", "How far the bus fell below open-circuit, and the inverter headroom it cost."],
   ["M10", "battery → inverter", "d_i_dc, E_bus_unserved", "Pack current delivered minus inverter current requested."]],
  [0.45 * inch, 1.55 * inch, 1.85 * inch, 3.05 * inch])
N("<b>Caution found by testing.</b> The electrical \"requested power\" <i>u_cmd · i</i> mixes real power with reactive cross-coupling voltage, and the current responds to what was <i>delivered</i>, so it is a counterfactual and can be negative. "
  "The shaft-power version (<i>T_cmd ω</i> against <i>T_elec ω</i>) is unambiguous and is the one to use. <b>Also:</b> a governor that lowers the current request makes M4/M5 look small while the speed shortfall (M1) is large. "
  "Mismatch depends on where you measure, so M1 is the complete number.")

# ======================================================================================================== CYCLES + DATA
H1("8. Real data: the TUM Tesla Model 3 dataset and cycles.py")
H2("8.1 The dataset")
B("<b>Electric powertrain data of a Tesla Model 3 during real-world driving situations</b> (Rosenberger et al., TUM, 2024; CC BY 4.0; doi:10.14459/2024mp1735471; paper doi:10.3390/wevj15060268). "
  "228 measurements recorded from the vehicle's CAN bus: high-voltage battery (current, voltage, temperatures, state of charge), powertrain (rotation, torque, temperature) and vehicle (speed, mileage). 6.3 GB as CSV and MAT files.")
B("<b>What we actually did.</b> The share allows listing and per-file download. We downloaded five tracks (0, 1, 10, 100, 103; 1.3 to 10.6 MB), verified each against the published SHA-512 checksums (all five match), and ran them. We did not download the other 223.")
H2("8.2 What the real data taught us (all measured on the files)")
L(["<b>Time</b> is epoch milliseconds (values near 1.6e12), about 10 Hz.",
   "<b>Schemas differ between tracks.</b> Tracks 10, 100 and 103 have no <font face='DV-M'>engine_rpm</font> column; every optional channel is looked up by name.",
   "<b>Column names mislead.</b> <font face='DV-M'>engine_rpm</font> is the <i>wheel</i> rpm: speed divided by it implies a 0.331 m radius, which matches a Model 3 tyre (a motor-rpm reading implies 3 m, which is absurd).",
   "<b>Battery current sign.</b> Discharge is negative: current correlates −0.86 with vehicle kinetic power.",
   "<b>rear_motor_torque is not trusted.</b> It is non-negative and correlates −0.46 with kinetic power, so its meaning is unclear. We do not use it.",
   "<b>Speed has gaps.</b> Track 1's GPS speed has 885 missing samples (4.9 %), interpolated and reported; 10 isolated spikes removed.",
   "<b>GPS elevation glitches.</b> Naive differentiation gave a 24.5 % grade; a median filter, a 30 s window and a ±15 % cap give 0.5° to 5.6°."])
H2("8.3 cycles.py: handling bad data on purpose")
B("Every anomaly is handled one of three ways and recorded in a report that is saved next to each run: <b>FIX</b> (repaired and noted), <b>ERROR</b> (refused with a named error and remedy), or <b>WARN</b> (accepted but suspicious). Nothing is repaired silently. "
  "51 automated cases cover the list below; writing them found six bugs in our own loader (section 10).")
T([["Group", "Cases covered (each is a test)"],
   ["File", "missing; directory; empty; only comments; UTF-8 BOM; Latin-1 bytes; ragged rows; title lines before the header; semicolon, tab, pipe and whitespace delimiters; decimal commas; 400,000-row file under 15 s"],
   ["Columns and units", "ambiguous or missing columns; mph, km/h and m/s from header or name; unitless values guessed and flagged; unknown unit refused; implausible speed (wrong units) refused; text such as \"12.3 m/s\" or \"n/a\" coerced"],
   ["Time", "epoch milliseconds; explicit time unit; out of order; duplicate timestamps; large gaps (keep longest stretch, or refuse); too short; irregular sampling resampled to a uniform grid; all-missing time"],
   ["Speed", "missing samples interpolated; mostly missing refused; all missing refused; small negatives clipped; reversing refused; isolated spikes removed; infinite values; constant speed and standstill warned"],
   ["Policy", "strict mode turns warnings into errors; arrays accepted; identical input gives identical output; extra measured channels aligned to the grid"],
   ["Vehicle fit", "gentle cycle fits; aggressive cycle refused with a remedy; motionless cycle refused; synthetic TUM schema variants (optional channels absent, GPS speed missing, no time column); five real tracks sane"]],
  [1.1 * inch, 5.8 * inch])
H2("8.4 Putting a real vehicle on a small motor")
B("The Tier 1 motor makes at most about 74 N·m (<i>1.5 p ψ</i> × 250 A with <i>i_d = 0</i>) and should stay below about 195 rad/s. A real car needs far more of both. "
  "<font face='DV-M'>fit_vehicle_to_motor</font> picks a gear ratio so the cycle's peak speed maps to 190 rad/s, then the largest <b>equivalent mass</b> whose torque demand stays under 85 % of capability for 99 % of the cycle. "
  "Road load (rolling, drag) and grade are reflected to the shaft as <i>m g C_rr/k</i>, <i>½ρ C_dA /k³</i> and <i>m g sinθ/k</i>, and inertia as <i>m/k²</i>, where <i>k</i> is motor speed per vehicle speed.")
T([["Track", "Length", "Peak speed", "Fits the motor?", "Equivalent mass"],
   ["track_0", "283 s", "23.8 m/s", "Yes", "130 kg"],
   ["track_10", "841 s", "28.2 m/s", "Yes", "102 kg"],
   ["track_103", "365 s", "22.8 m/s", "Yes", "118 kg"],
   ["track_1", "1,796 s", "38.4 m/s", "No: drag exceeds the motor; fits only with road load scaled to 59 %", "20 kg (minimum)"],
   ["track_100", "293 s", "39.0 m/s", "No: fits only with road load scaled to 51 %", "20 kg (minimum)"]],
  [0.9 * inch, 0.8 * inch, 0.95 * inch, 3.15 * inch, 1.1 * inch])
B("Refusing the two highway tracks is the correct behaviour, not a failure: it is the same field-weakening limit that makes Tier 2 hard. The loader says so and offers an explicit, recorded scaling option.")
N("<b>Assumed.</b> The equivalent mass is a toy number, not the Tesla's 1,900 kg. The drag area (0.62 m²), rolling coefficient (0.010) and the 190 rad/s target are assumptions. "
  "EPA UDDS also loads cleanly (1,369 s, 25.3 m/s, fits at 287 kg) but has not yet been run through the powertrain.")

# ======================================================================================================== RESULTS
H1("9. What the experiments show")
H2("9.1 The 8 s scenarios on the plain motor")
B("Speed ramps to 200 rad/s, a 10 N·m grade load starts at 1.5 s, and at 4.5 s the voltage ceiling drops from 173 V to 25 V for one second (an unannounced event). Identical across runs so only the controller differs.")
T([["Run", "Controller", "Estimate ψ^ peak (true 0.066)", "Peak current", "Speed error rms after event", "Peak Δu"],
   ["A", "naive, no event", "0.066", "151 A", "0.0 rad/s", "0 V"],
   ["B", "naive", "2.741 Wb (runaway)", "589 A", "64.1 rad/s", "2,068 V"],
   ["C", "σ-mod + projection", "0.100 (bounded)", "255 A", "6.0 rad/s", "825 V"],
   ["D", "anti-windup", "0.0665", "250 A", "0.81 rad/s", "806 V"],
   ["Dg", "anti-windup + governor", "0.0662", "224 A", "0.07 rad/s", "706 V"]],
  [0.5 * inch, 1.45 * inch, 1.75 * inch, 0.9 * inch, 1.35 * inch, 0.95 * inch])
B("This reproduces the failure the notes predicted (B: the estimate runs away to 41 times its true value) and shows both proposed fixes work, with anti-windup better than the leak.")
H2("9.2 With the battery: the event is now physical")
B("The BMS cuts the allowed pack current to 5 A for one second at 4.5 s, below the 12.6 A cruise demand. The bus collapses to about 60 V. Unprotected: speed dips to 137 rad/s, reaches only 184 by the end, peak voltage shortfall 924 V, "
  "35.7 kJ of shaft energy owed. Protected (anti-windup, governor, sees the bus voltage): dips to 172, recovers to 200, peak shortfall 231 V, 0.1 kJ owed.")
H2("9.3 A real drive")
FIG("docs/figures/fig_realdrive.png", 6.7 * inch, "Figure 2. TUM track_0 (a real Tesla Model 3 drive) driving the simulated powertrain. The BMS cut is applied for 3 s at the moment of peak acceleration power (118.5 s).")
B("<b>Verified:</b> both 283 s runs completed (56,580 logged rows each, 1.4 million physics steps, about 400 s of compute), nothing diverged, every channel is finite. Away from the event the simulated speed follows the real one closely (tracking error 3 to 3.5 % of peak). "
  "During the event the unprotected controller falls 32 rad/s behind the request and the protected one 22; the protected run's bus voltage falls further (41 V against 176 V) because the controller is asking for less, and it still tracks better afterwards.")
H2("9.4 How well does the simulated battery resemble the real one?")
FIG("docs/figures/fig_realvsim.png", 6.7 * inch, "Figure 3. Battery power during the same drive. The real trace is scaled by the mass ratio 0.069 (equivalent mass over 1,900 kg).")
L(["<b>Agreement is moderate:</b> correlation 0.40 at 0.1 s, 0.52 at 1 s, 0.65 at 5 s smoothing.",
   "<b>Regeneration disagrees:</b> real 38 % of drive energy, simulated 12 %. The toy vehicle and the simple deceleration behaviour are likely causes; not yet investigated.",
   "<b>Net energy disagrees:</b> 0.035 kWh (real, scaled) against 0.116 kWh simulated.",
   "<b>Pack resistance:</b> a fit to the real data gives about 0.062 Ω against our 0.085 Ω.",
   "<b>What this does and does not show.</b> It shows the pipeline runs end to end on real data and gives a first calibration target. It does not show the simulated powertrain predicts a real Tesla; the vehicle is a scaled toy."])

# ======================================================================================================== BUGS
H1("10. Mistakes we found in our own work")
B("These are listed because they show what the checks catch and where trust should be limited.")
T([["#", "What went wrong", "How it was found", "Fix"],
   ["1", "Adaptive law had the wrong sign; the estimate diverged to −6.8 Wb with no saturation", "Baseline run", "Re-derived; warning left in the code"],
   ["2", "Scenario at 300 rad/s was infeasible (cross-coupling 216 V > 173 V)", "Voltage ellipse probe", "Lowered to 200 rad/s"],
   ["3", "After the event the drive sat in a trap (230 A, 18 N·m)", "Experiments C and D", "Added the governor"],
   ["4", "np.cross on 2-vectors fails in NumPy 2", "Unit test", "Explicit 2-D cross product"],
   ["5", "Simulator clipped silently up to 4 V when regeneration pushed the bus above 300 V", "The viewer's own M3 check showed 3.7 V", "Capped the ceiling; regression test"],
   ["6", "Electrical \"requested power\" came out negative", "Cumulative energy looked wrong", "Added the shaft-power mismatch; caveat documented"],
   ["7", "A 25 A BMS derate never bound (cruise draws 12.6 A), so no event occurred", "Run summary showed no sag", "Derate set to 5 A"],
   ["8", "utf-16 decoding accepted arbitrary bytes as garbage text", "Test 06", "utf-16 only with a byte-order mark"],
   ["9", "Unit names like speed_mph were not recognised (underscore is a word character)", "Test 13", "Lookaround regular expressions"],
   ["10", "Read-only arrays crashed spike handling", "Test 34", "Copy on load"],
   ["11", "A dirty data row (\"n/a\") was taken as the header", "Test 18", "Header = nearest all-text row"],
   ["12", "Time unit inferred from unsorted time (a 0.3 s cycle)", "Test 21", "Infer from sorted differences"],
   ["13", "Naive GPS elevation differentiation gave a 24.5 % grade", "Real track 1", "Median filter, 30 s window, cap"],
   ["14", "Full-rate validation checks showed misleading \"pass\" or large errors on decimated real runs", "Viewer on the real drive", "Those checks now show SKIP at low logging rates"]],
  [0.3 * inch, 2.75 * inch, 1.9 * inch, 1.95 * inch])

# ======================================================================================================== ASSUMPTIONS
H1("11. Assumptions, limits and questions for a reviewer")
H2("11.1 Not yet true (limits)")
L(["<b>No stability proof.</b> V-dot ≤ −λV + c is unverified. The logged V(t) is a candidate only.",
   "<b>One unknown parameter (ψ) and one kind of event</b> per run. No noise, no sensor delay, no parameter drift yet.",
   "<b>Tier 1 voltage ceiling is capped at 173.2 V</b>; real field weakening is not implemented.",
   "<b>The vehicle is a toy:</b> equivalent mass, assumed drag and rolling coefficients, assumed gear ratio.",
   "<b>Battery numbers are generic</b> (OCV table, resistances, BMS limits, inverter loss, thermal mass). A real pack would differ.",
   "<b>Five of 228 real tracks</b> were used. No other dataset (for example Argonne D3, which blocked automated download) has been opened.",
   "<b>Full-rate checks need full-rate logs.</b> The 283 s runs were logged at 200 Hz, so those validation rows are skipped.",
   "<b>Sensor noise and a cycle-dataset sweep are the next build</b>, not done."])
H2("11.2 Good questions to put to a reviewer")
L(["Is the sign and form of <i>dψ^/dt = −Γ ω_el e_a</i> correct, and does the compensated error <i>e_a = e − z</i> really recover the unsaturated dynamics for this loop?",
   "Is leaking toward the <i>nominal</i> value (not zero) a legitimate σ-modification for a parameter far from zero, and does it keep the proof's inequality?",
   "Is a radial projection onto the disc of radius <i>V_dc/√3</i> an acceptable saturation model for a BIBS or UUB theorem? What does it assume about the actuator?",
   "Does the implicit bus solve with an explicit constant-power current (using the previous <i>V_dc</i>) risk a spurious oscillation near collapse?",
   "The governor lowers the current request, which hides the shortfall at M4 and M5. Which mismatch should the proof be stated in terms of?",
   "Is the equivalent-mass scaling a defensible way to run a real vehicle trace on a smaller motor, or should we scale power instead?",
   "What would a fair calibration of the pack resistance from the real trace look like, given the vehicle is scaled?"])

# ======================================================================================================== GLOSSARY
H1("12. Glossary")
T([["Term", "Meaning here"],
   ["PMSM", "Permanent-magnet synchronous motor: the motor type used in most EVs."],
   ["dq frame", "Coordinates rotating with the rotor; turns three phase quantities into two DC-like ones."],
   ["Back-EMF", "Voltage the spinning magnet generates; grows with speed (ω_el ψ)."],
   ["Field weakening", "Injecting negative i_d to reduce flux so the motor can spin faster than the voltage budget otherwise allows."],
   ["Inverter / duty cycle", "Power electronics that turn DC bus voltage into three phase voltages; duty cycle is each leg's on-time fraction."],
   ["DC link / bus", "The capacitor-smoothed DC voltage between battery and inverter."],
   ["BMS", "Battery management system; limits pack current and voltage."],
   ["OCV, SOC", "Open-circuit voltage and state of charge of the pack."],
   ["Adaptive control", "A controller that estimates an unknown parameter online."],
   ["Windup", "An integrator or estimator accumulating error the actuator cannot act on."],
   ["σ-modification, projection", "Leakage that pulls an estimate back, and a hard clamp that keeps it in a physical range."],
   ["Anti-windup", "Extra state that tracks the actuator's shortfall so the controller can discount it."],
   ["Reference governor", "A filter that stops the reference asking for something the system cannot do."],
   ["Lyapunov function V", "An energy-like function; if it cannot grow, the state stays bounded."],
   ["BIBS, UUB", "Bounded-input bounded-state stability; uniform ultimate boundedness (states end inside a calculable bound)."],
   ["Equivalent mass", "The vehicle mass, reflected to the motor, that the toy motor can actually accelerate."]],
  [1.5 * inch, 5.4 * inch])

H1("13. Sources")
L(["Rosenberger N. et al. (2024). Electric powertrain data of a Tesla Model 3 during real-world driving situations. TUM mediaTUM. https://doi.org/10.14459/2024mp1735471 (CC BY 4.0). Related paper: https://doi.org/10.3390/wevj15060268",
   "U.S. EPA Urban Dynamometer Driving Schedule (UDDS) speed trace, EPA dynamometer drive schedules.",
   "Gym-Electric-Motor 3.0.3 (Cont-CC-PMSM-v0) and FASTSim 3.1.0 open-source packages.",
   "Pozzato G. et al. Real-world electric vehicle data: driving and charging. Mendeley Data. https://doi.org/10.17632/7vdkzpnjgj.2 (identified as a candidate; its files were not downloaded)."])

def footer(c, d):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(MUTED)
    c.drawString(0.75 * inch, 0.45 * inch, "EV powertrain simulation explainer")
    c.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch, f"Page {d.page}"); c.restoreState()

doc = BaseDocTemplate("docs/Powertrain_Sim_Explainer.pdf", pagesize=letter, leftMargin=.75 * inch, rightMargin=.75 * inch, topMargin=.7 * inch,
                      bottomMargin=.75 * inch, title="EV Powertrain Simulation: what we built and why", author="Rudolph")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=footer)])
doc.build(S)
print("built")
