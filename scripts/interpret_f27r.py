"""Interpret f27r decoded text as a medical recipe using Sonnet."""
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

lines_heb = [
    ["swnt", "SpEn", "wpJwEn", "wlk", "bhyh", "SEn", "rwEn", "Shyk"],
    ["Sr", "brywA", "mwEn", "bryr", "syr", "rStwEn", "mwkr", "SpEJA", "nr"],
    ["mwk", "SEn", "mwpt", "mwk", "SEn", "mwEn", "Sk", "bhyr", "Spk", "gyr"],
    ["Spt", "swk", "syk", "Sk", "Skr", "Splt", "nwk", "SrwEt"],
    ["swr", "nplk", "wEtb", "SEn", "ShyEl", "bhyr", "srlr"],
    ["Sk", "nmwkt", "swk", "mwk", "bhyk", "SEn", "mykt", "Sr"],
    ["Spkt", "Spk", "mwpkt", "Skl", "Spkn", "Sm", "nmyk", "gyk"],
    ["SkJS", "Sk", "mwkJ", "Sr", "SpkJ", "bryr", "mwk", "myr"],
    ["Spkr", "rwppt", "wEn", "Spkt", "mwk", "SJw", "Sk", "gmrt"],
    ["Spkt", "npXt", "n", "S", "Sk", "Spkt", "Sr"],
    ["mwk", "brhyt", "nppk", "nwk", "SpJA", "byr"],
    ["bryr", "Spptwpk", "Spk", "SpEth", "mytq"],
    ["Spprwktw"],
]

prompt = """You are an expert in medieval Hebrew medical manuscripts and Judeo-Italian medical tradition.

Below is a decoded page from the Voynich manuscript (folio f27r, herbal section). The text is in CONSONANTAL HEBREW (no vowels) written by a 15th-century Italian Jewish physician.

HEBREW ASCII NOTATION:
A=aleph b=bet g=gimel d=dalet h=he w=vav z=zayin X=chet J=tet y=yod
k=kaf l=lamed m=mem n=nun s=samekh E=ayin p=pe C=tsade q=qof r=resh S=shin t=tav

IMPORTANT CONTEXT:
- This is a HERBAL page with a plant illustration
- The text likely describes: plant identification, properties, preparation, medical uses
- Key repeated words: mwk(x6), Sk(x6), SEn(x5), Spk(x3) = the formulaic vocabulary
- mwk can mean "cotton/soft material" in medical context, not just "be poor"
- SEn can mean "to apply/rest upon" (applying medicine), not just "to lean"
- Spk = to pour -- recipe instruction
- swk = to anoint -- recipe instruction
- gyr = chalk/ite -- ingredient
- bhyr = bright/clear -- describing appearance
- Skr = strong drink/alcohol -- solvent
- nr = lamp/light
- Scribal note: t and J (tav/tet) are used interchangeably
- Scribal note: X and E (chet/ayin) may be confused
- Scribal note: w and r (vav/resh) may be confused

DECODED TEXT (line by line, Hebrew consonantal):
"""

for i, line in enumerate(lines_heb):
    prompt += f"Line {i+1}: {' '.join(line)}\n"

prompt += """
TASK:
1. For each line, provide a plausible vocalization and Italian/English translation
2. Try to read this as a COHERENT medical recipe or herbal description
3. Note which words you are confident about vs uncertain
4. Consider alternate readings where the standard dictionary gloss does not fit the medical context
5. Be honest about what you can and cannot read

Format: for each line, show the consonantal text, your proposed reading, and translation."""

client = anthropic.Anthropic(api_key=api_key)
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4000,
    messages=[{"role": "user", "content": prompt}],
)
print(msg.content[0].text)
