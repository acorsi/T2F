#!/usr/bin/env python3
"""
Interface graphique pour configurer et lancer la simulation T2F.
À placer à la racine du projet (même niveau que simulation/).
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
from pathlib import Path

ROOT        = Path(__file__).parent
CONFIG_FILE = ROOT / "simulation" / "config.txt"
BUILD_DIR   = ROOT / "simulation" / "build"
BINARY      = BUILD_DIR / "T2F"


def save_config(fields):
    lines = [
        f"sequential {'yes' if fields['sequential'].get() else 'no'}",
        f"parallel {fields['parallel'].get()}",
        f"events {fields['events'].get()}",
        f"elossmodel {fields['elossmodel'].get()}",
        f"elosstreat {fields['elosstreat'].get()}",
        f"pad {fields['pad'].get()}",
        f"pressure {fields['pressure'].get()}",
        f"skipgefline {fields['skipgefline'].get()}",
    ]
    CONFIG_FILE.write_text("\n".join(lines) + "\n")


def run_pipeline(fields, log):
    def append(msg):
        log.configure(state="normal")
        log.insert(tk.END, msg)
        log.see(tk.END)
        log.configure(state="disabled")

    save_config(fields)
    append("✓ config.txt sauvegardé\n")

    append("⟳ Compilation en cours...\n")
    result = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        append(f"✗ Erreur de compilation :\n{result.stderr}\n")
        return
    append("✓ Compilation réussie\n\n")

    append(f"▶ Lancement de {BINARY.name}...\n")
    proc = subprocess.Popen(
        [str(BINARY)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        cwd=str(BUILD_DIR)
    )
    for line in proc.stdout:
        append(line)
    proc.wait()
    if proc.returncode == 0:
        append("\n✓ Simulation terminée.\n")
    else:
        append(f"\n✗ Le programme s'est terminé avec le code {proc.returncode}.\n")


def launch(fields, log, btn_launch):
    btn_launch.configure(state="disabled")
    log.configure(state="normal")
    log.delete("1.0", tk.END)
    log.configure(state="disabled")
    t = threading.Thread(target=lambda: run_pipeline(fields, log), daemon=True)
    t.start()
    def wait():
        t.join()
        btn_launch.configure(state="normal")
    threading.Thread(target=wait, daemon=True).start()


def main():
    root = tk.Tk()
    root.title("T2F — Simulation")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use("aqua")

    FONT = ("Helvetica", 13)

    # Titre
    tk.Label(root, text="T2F  —  Simulation TPC",
             font=("Helvetica", 14, "bold"), anchor="w").grid(
        row=0, column=0, columnspan=2, pady=(16, 8), padx=20, sticky="w")

    # Cadre paramètres
    frame = ttk.Frame(root, padding=(20, 0, 20, 10))
    frame.grid(row=1, column=0, columnspan=2, sticky="ew")

    fields = {}

    def add_row(label, widget_fn, r):
        tk.Label(frame, text=label, font=FONT, anchor="e", width=17).grid(
            row=r, column=0, sticky="e", pady=3, padx=(0, 10))
        w = widget_fn(frame)
        w.grid(row=r, column=1, sticky="w", pady=3)
        return w

    # sequential
    seq_var = tk.BooleanVar(value=False)
    fields["sequential"] = seq_var
    tk.Label(frame, text="sequential", font=FONT, anchor="e", width=17).grid(
        row=0, column=0, sticky="e", pady=3, padx=(0, 10))
    ttk.Checkbutton(frame, variable=seq_var).grid(row=0, column=1, sticky="w", pady=3)

    # parallel
    par_var = tk.StringVar(value="14")
    fields["parallel"] = par_var
    add_row("parallel", lambda f: ttk.Entry(f, textvariable=par_var, width=10, font=FONT), 1)

    # events
    ev_var = tk.StringVar(value="1000000")
    fields["events"] = ev_var
    add_row("events", lambda f: ttk.Entry(f, textvariable=ev_var, width=12, font=FONT), 2)

    # elossmodel
    el_var = tk.StringVar(value="G4")
    fields["elossmodel"] = el_var
    add_row("elossmodel",
            lambda f: ttk.Combobox(f, textvariable=el_var,
                                   values=["G4", "LG", "SRIM"],
                                   state="readonly", width=8, font=FONT), 3)

    # elosstreat
    et_var = tk.StringVar(value="svd")
    fields["elosstreat"] = et_var
    add_row("elosstreat",
            lambda f: ttk.Combobox(f, textvariable=et_var,
                                   values=["native", "svd"],
                                   state="readonly", width=8, font=FONT), 4)

    # pad
    pad_var = tk.StringVar(value="2")
    fields["pad"] = pad_var
    add_row("pad (mm)", lambda f: ttk.Entry(f, textvariable=pad_var, width=8, font=FONT), 5)

    # pressure
    pres_var = tk.StringVar(value="300")
    fields["pressure"] = pres_var
    add_row("pressure (mbar)", lambda f: ttk.Entry(f, textvariable=pres_var, width=8, font=FONT), 6)

    
    # skipgefline
    skip_var = tk.StringVar(value="0")
    fields["skipgefline"] = skip_var
    add_row("skipgefline",
            lambda f: ttk.Spinbox(f, textvariable=skip_var,
                                  from_=0, to=10_000_000, increment=100_000,
                                  width=10, font=FONT), 7)

    # Log
    ttk.Separator(root).grid(row=2, column=0, columnspan=2,
                              sticky="ew", padx=20, pady=(4, 8))

    log = scrolledtext.ScrolledText(
        root, width=70, height=16, state="disabled",
        font=("Menlo", 10), relief="sunken", bd=1
    )
    log.grid(row=3, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")

    # Boutons
    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 16), sticky="e")

    ttk.Button(btn_frame, text="Quit", command=root.destroy).pack(
        side="left", padx=(0, 10))

    btn_launch = ttk.Button(btn_frame, text="▶  Lancer",
                            command=lambda: launch(fields, log, btn_launch))
    btn_launch.pack(side="left")

    root.mainloop()


if __name__ == "__main__":
    main()
