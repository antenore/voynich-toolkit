"""Batch interpret f45r + f15v + f99r with merged vocabulary from f27r/f75r."""
import anthropic
from pathlib import Path

api_key = ""
for env_path in [Path(__file__).parent.parent / ".env", Path(__file__).parent.parent.parent / ".env"]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "ANTHROPIC_API" in line and "=" in line:
                k, v = line.split("=", 1)
                api_key = v.strip().strip('"').strip("'")
                break

VOCAB = """CONFIRMED VOCABULARY (from f27r herbal + f75r balneological):
- mwk = cotton/soft filtering material (NOT "be poor")
- SEn = to rest/apply medicine (NOT "to lean")
- Spk = to pour (recipe instruction)
- swk = to anoint
- Sk = mixture/preparation
- bhyr = bright/clear/limpid (liquid quality)
- Spt = to pour/set
- gyr = chalk/ite (clarifier ingredient)
- Skr = strong drink/alcohol (solvent)
- mr = bitter (indicator/quality)
- mytq = sweet (completion taste test)
- gmrt = completed (finishing step)
- nr = lamp/light
- mwpt = sign/wonder (diagnostic)
- SrpEn = bathing/soaking (balneological key term)
- Srppt = immersion cycle
- ryJ = warm/heating
- ryt b = "in/within" (preposition formula)
- wdryn = roses (vered)
- syr = liquid? (frequent, needs confirmation)
- bryr/bryt = clear/pure

SCRIBAL NOTES: t/J interchangeable, X/E may be confused, w/r may be confused."""

pages = {
    "f45r": {
        "section": "herbal",
        "lines": """L01: myrStSl SEtryEn Stw SpXn l syk myklr syklS syklw
L02: swEn SEJA St mwt bhyt mwk mwk bhytS syr gw
L03: SEJAk mwkt bhyr bhy SEJA bhyk SpptS gwEJA
L04: Shwm bhyt mwptS mwptS gyk
L05: mwt wEn swkl Spkt SJ Srmyklw SEnJw Skt mwr bhyr
L06: sykJr sykJr bhyr SrwkJS mwmwkJ syJr myr Shyr
L07: sykm s S syt syr
L08: mwkt SEJA bhyr SEJA ryr b SpEJA SJr myr bhyr g
L09: swr sw bryk ryr b SrwmwtS bhyJS mwkt mwt Sr
L10: bhytr syEn StS Str sryt Srmyr swmyr mwpk myr
L11: m bhytw wktw yrmym b bhyrmyr""",
    },
    "f15v": {
        "section": "herbal",
        "lines": """L01: swhwl SEnhr bhwk bykJr swklw Sr
L02: swk sw whw s bhy SEJA t bry syr
L03: swEJA bhyr swt swptw bhytw
L04: bhwr Stwk mwEn St rwkJ
L05: swkJr swk swk swkJS SEJA n
L06: SpkJ SJwk bhyt SkJw s bhy
L07: A Sr bhwk wEn n Sk n Sk swJ nmr
L08: swkJS swk mr bhw SJw mwEn bhyr
L09: mwkJw w mwEJA mwk mwk Srwk byt
L10: swn swk bhwEJA SEJA bhyt
L11: bhwmwn swpk mwk bhyr SEJA
L12: bhyr swEJA mwk swk""",
    },
    "f99r": {
        "section": "pharmaceutical",
        "lines": """L01-14 (labels): Xwgyhwtw | sytw S | syhyr | Stw | wmyn | whw | bry | swtw | mwmyn | mypptn | Shytw | wmwtw | gymyJw | SrmwJw
L15: Srwpkl SrwpJw bhyr SppElA mwEnt myr mwpElA bhymr syh
L16: SEnwhyh swEn swEnn mwpEn mwt mwpEn mwp mwEn mwt mwEtA ShwEtA
L17: swpkr nk my SEtAS mwptw swEtA bhyhw swk swppt Shwk
L18: swpt mwk mwtS Sppk Srwk mwEtA bhyr Smwptw bhyr SEtA
L19-27 (labels): lw yhy m | whyr | Smwmy | swhww | Stwk | Stw | Smwptw | rmwpJS | SkJwpk
L28: bhyrwJ Spklw St swr swkJw wEnlw mwptw mwpEn mwlpwpJw
L29: mwpEnr SpEtA mwpEtA mwptw mwpEJA Spt mwEtA mwptw Spptw rmyr
L30: mwJ Srwpk mwt Smwtw Srmwtw Smwt myt mwktw rmwt
L31: nppk Spptw mwJ mwpEn bhyr mwJ mwptw
L32-40 (labels): SrmwEnJ | nwtw | Sptpw | swr | SEEtAwk | s Spk Jr | Sdhyn | Shwh | gyr
L41: swppEnJ mwpElA SptA mwkl SpEtA mwklS swk Smwk SpEJAwJ Shwt
L42: mwn mwpEn SpEnpt Spppt nk Spk mwr Spk SppEJA bhyr gwpk
L43: bhyr Spptwppk SpEtApk swr Srmw SppEn Srwpt Sppptw n bhy nmw
L44: Spt Spptk Spk SpEtA SpEtA SpptS bhw sry Srwk yntpw
L48: mwptS mwptw w wpEtA mwk myrwpk wptw s gppkmw syhw
L49: Sppptw Sppt swppt Spptw bhyr mwptw n bhy shymw w mynmw
L50: wppt Spptw Sppt Snptw wppt syn StpnpEn sw my
L51: bryEnS S SpEt SpEJAw Sr bhyr swtr Spptw JAEn S En
L52: swkS nmr sw gy sry gw""",
    },
}

client = anthropic.Anthropic(api_key=api_key)

for folio, info in pages.items():
    print(f"\n{'='*70}")
    print(f"  {folio} — {info['section']}")
    print(f"{'='*70}\n")

    prompt = f"""You are an expert in medieval Hebrew medical manuscripts (Judeo-Italian tradition, 15th century).

{VOCAB}

FOLIO: {folio} — Section: {info['section']}
{"This is a PHARMACEUTICAL page with ingredient labels and recipe text." if info['section'] == 'pharmaceutical' else "This is a HERBAL page describing a plant and its medicinal preparation."}

DECODED TEXT:
{info['lines']}

TASK:
1. Read this as a COHERENT medical text using the confirmed vocabulary above
2. Identify NEW words not in the vocabulary (especially plant names, body parts, conditions)
3. Give a flowing readable translation (not word-by-word)
4. Note key discoveries (plant names, new verbs, ingredients)
5. Keep it concise — focus on what the text SAYS, not linguistic analysis"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    print(msg.content[0].text)
    print()
