import re
import sys

def oprav_vlnky_pro(text):
    # 1. Jednopísmenné předložky a spojky (v, z, k, s, u, o, a, i)
    text = re.sub(r'(?<=\s)([iaouksvzIAOUKSVZ])\s+', r'\1~', text)
    
    # 2. Vícepísmenné předložky (přes, bez, beze, mimo, podle, stran, kolem...)
    # Přidáváme vlnku za vybrané časté předložky
    predlozky = ['přes', 'bez', 'beze', 'mimo', 'podle', 'kolem', 'stran', 'včetně']
    for p in predlozky:
        # Regex pro zachycení předložky jako celého slova
        pattern = rf'(?<=[\s~])({p})\s+'
        text = re.sub(pattern, r'\1~', text, flags=re.IGNORECASE)

    # 3. Čísla následovaná jednotkou nebo symbolem (např. 10 kg, 5 %, 200 km)
    # Hledá číslo následované mezerou a pak písmenem nebo znakem %
    text = re.sub(r'(\d+)\s+([a-zA-Z%]+)', r'\1~\2', text)
    
    return text

def zpracuj_soubor(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            obsah = f.read()
        
        opraveno = oprav_vlnky_pro(obsah)
        
        output_file = "vlnka_" + input_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(opraveno)
            
        print(f"Hotovo! Opravený text je v: {output_file}")
    except Exception as e:
        print(f"Chyba: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Použití: python skript.py soubor.tex")
    else:
        zpracuj_soubor(sys.argv[1])