"""Interpret f75r decoded text as a balneological/medical recipe using Sonnet.
Uses vocabulary learned from f27r as bootstrap context."""
import anthropic
from pathlib import Path

# Load API key
api_key = ""
for env_path in [Path(__file__).parent.parent / ".env", Path(__file__).parent.parent.parent / ".env"]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "ANTHROPIC_API" in line and "=" in line:
                k, v = line.split("=", 1)
                api_key = v.strip().strip('"').strip("'")
                break

# f75r decoded lines (from crib attack + decode)
lines_text = """L01: t Srpk Shyt Spptw syt SEn t Srpk syJ SrpEn
L02: bryr SpEn Sm pEnn mw m Srpk sytSrpk Srpptpk swh
L03: bryt myk sw Spk Sp ryt b StppEn ryJm b sytmw sw
L04: SEtAyr wgytm SpptS SpEnm myt Sr SpEn sw SpEn Srppt
L05: SpXn syt Spk SpEtA s bry mr mr SrppEn Sppt St
L06: l Spk swppt Stmr syr Sptw bryt SEJAk Srppt St
L07: l Srpk SrEnt ryJS b Srpk syt Sk mwm Srpk St
L08: n sw Spk SrhyJ SEtApEnr bryt SEtAk SrpEnm Srpptw
L09: Srkt SEJAk wm SrpJ byt SEtApk syt mr Srpk myn
L10: swXnr syJ Srk SpXn bryt SEtAk Sw SpJw SrpJ m Srpk
L11: Srppt ryt b Smw Srppt Sr myt sytw SrpEn swr gytpk
L12: SEtApEnn myt Smr SpEn s mw Sppk SpEn Sw mw SrpEn St
L13: l Srpk Srppt Srpt SrpJ SrpJ SrpJ bryt SrpEnmr
L14: ryn b mr SEnppt ryt b Sr SrpEnmr bryt SEtAk bryt syJr Smy
L15: n ry b bryt m Smwppt bhyn Srpk mwn sw SrpEn Srktr St
L16: SrpEnr syt SrppEn m k wEn ryt b Sk SrpEnJw ryt b Srpk
L17: l syrpk SklpEn ShyrpEnm myr myEn SEn mwt SrpEn gyt
L18: n mr SrppEn m SrpEn m brytw syk n sy Smr
L19: SrpEnt m SpEn St SpEn SpEJr bryt sy
L20: Sh SpEn s Spk Spkm wm ryrS b SpEn ryt b
L21: rytp?w b Spk SnEtA sw wpEnS s mr swm gy
L22: syrw SpEn ryt b Srpk sw SpEn yt s Srpk syn
L23: l Spk SpEnt Sppt myt SpEnn m Srpk gyt
L24: ryt b Spptmw sytm S ryn b SEJApk swm mw
L25: bhyn EtAEn S Srppt SEn Sr Srppt Srpkm gyh
L26: ryr b mw mwpEn ryr b mr Sm syr Sry
L27: l swEnmyr mwtEn SJ syEnl SrpEn Srmytr syr sytw SrpJw Sr mwh
L28: SrpkJ Srpkl Sppt mwn mr sryr SEJApEn m m SrppEn Srppt mwm Srpk
L29: bryr mytk Sr wm bhytm mw Srpptw m ryr b Spkmw Srppt Srpk byJ
L30: bhyr Stppk Spt ryt b Stpk myt bryr Srpk mwmytw SrpEn sytw wmr g
L31: SrpEnw Spt SEtAk syt SrpEn Spk rpt Srpt Srmyr
L32: l SEnmr myr SrpEn bryt nyr Srnk SrpEn syt SrpEn Srm
L33: Sppt m SrpEn m Srpk ryt b SrpEJAk SrpJm syr w g
L34: n Srptmw mytw syr SJw syJw syJw mw bryt Srptmw
L35: ryt b SJppEn ryt b syr syr SrpEn yt s mr Sr
L36: mwn Srppt Srppt Spt sytw syJw syr syr Sr
L37: Srpt Sr SJppEn Srpt Srkt Srkpt mwm
L38: Srppt Srppt Srpt Srpt Srppt Srm
L39: SrpEnS Srppt Srkt Srpptmw SpJw Srmwt
L40: yr s SrpEn ryt b SrpEn myr Srppt s SrpEn
L41: n Srpptw Srppt SrppJw St SrpptSr n S
L42: SrpEnr Srpt A SppJ SrppJ syr
L43: SrpEnS sytpk Srmw ryt b sytk syJw Srmw
L44: Srpkr ryn b Srptw Srpt SrpJr Srmwtr syJr yJr g bhymr k syr Sr
L45: SrpEnn SEtAEn Spt Srptr mwhwn SJr SrpJr SrpJr SrpJw bhytw
L46: Spt Srpt mwpEn Srpt ryr b SrpEn mwJw Srpk swmw
L47: myn Srpptw
L48: myr S SpkS
L49: nmwn whyr
L50: StkS
L51: wdryn
L52: Srmyn
L53: Sdryr"""

prompt = f"""You are an expert in medieval Hebrew medical manuscripts and Judeo-Italian balneological tradition (bathing/hydrotherapy treatments).

Below is a decoded page from the Voynich manuscript (folio f75r, BALNEOLOGICAL section — one of the famous "bathing" pages showing figures in pools/tubs). The text is in CONSONANTAL HEBREW (no vowels) written by a 15th-century Italian Jewish physician.

HEBREW ASCII NOTATION:
A=aleph b=bet g=gimel d=dalet h=he w=vav z=zayin X=chet J=tet y=yod
k=kaf l=lamed m=mem n=nun s=samekh E=ayin p=pe C=tsade q=qof r=resh S=shin t=tav

VOCABULARY FROM f27r (CONFIRMED — use as anchors):
- mwk = cotton/soft material (for filtering, bandaging) — NOT "be poor"
- SEn = to rest/apply (medicine) — NOT just "to lean"
- Spk = to pour (recipe instruction)
- swk = to anoint
- gyr = chalk/ite (ingredient/clarifier)
- bhyr = bright/clear (describing solution/liquid)
- Skr = strong drink/alcohol (solvent)
- nr = lamp/light
- Sk = mixture/preparation
- Spt = to pour/set
- mwpt = sign/wonder (diagnostic)
- mytq = sweet (taste test)
- gmrt = completed (finishing step)

SCRIBAL NOTES:
- t and J (tav/tet) are interchangeable
- X and E (chet/ayin) may be confused
- w and r (vav/resh) may be confused

THIS IS A BALNEOLOGICAL PAGE, so expect vocabulary about:
- Bathing, immersion, water temperature
- Herbal baths, medicinal soaks
- Body parts, skin conditions
- Heating/cooling water
- Duration, timing, repetition

DECODED TEXT (53 lines):
{lines_text}

TASK:
1. This is a LONG text (488 words, 53 lines). Focus on the FIRST 20 LINES in detail.
2. For lines 21-53, give a summary interpretation.
3. Identify the main RECIPE STRUCTURE: what is being prepared, how, for what condition
4. List NEW vocabulary you can identify (words not in the f27r vocabulary above)
5. Be honest about confidence levels.
6. Try to identify recurring patterns/formulas.

IMPORTANT: This is a balneological text. Read it as bathing/hydrotherapy instructions, not just an herbal recipe."""

client = anthropic.Anthropic(api_key=api_key)
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=5000,
    messages=[{"role": "user", "content": prompt}],
)
print(msg.content[0].text)
