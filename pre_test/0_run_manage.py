import subprocess
import datetime

scripts = [
    "pre_test/f_BASECcode/0_get_features.py",
    "pre_test/f_Coherence/0_get_coherence.py",
    "pre_test/f_CWEcode/0_getCWE.py",
    "pre_test/f_PPLcode/1_getPPL.py"
]

for script in scripts:
    print(f"{datetime.datetime.now()}, Running: {script}")
    subprocess.run(["python", script], check=True)
