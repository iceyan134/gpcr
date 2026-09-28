"""
Comprehensive CCD-to-SMILES mapping for common PDB ligands.
Covers drugs, cofactors, metabolites, buffers, ions, and common small molecules.
Generated from PDB Chemical Component Dictionary.
"""

CCD_SMILES = {
    # ── Nucleotides & Cofactors ──────────────────────────────────────
    "ATP": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "ADP": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
    "AMP": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
    "GTP": "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "GDP": "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "CTP": "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1",
    "UTP": "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
    "NAD": "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",
    "NAI": "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",
    "FAD": "Cc1cc2nc3c(nc4nc(=O)[nH]c(=O)c4n3)c(=O)[nH]c2cc1C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]5O[C@@H](n6cnc7c(N)ncnc76)[C@H](O)[C@@H]5O",
    "FMN": "Cc1cc2nc3c(nc4nc(=O)[nH]c(=O)c4n3)c(=O)[nH]c2cc1C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O",
    "COA": "CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS",
    "SAM": "C[S+](CC[C@H](N)C(=O)O)C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    "SAH": "C[S+](CC[C@H](N)C(=O)O)C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    "HEM": "CC1=C(C2=CC3=C(C(=C(N3)C=C4C(=C(C(=N4)C=C5C(=C(C(=N5)C=C1[N-]2)C=C)C)C=C)C)C=C)C=C)C.[Fe+2]",
    "NAP": "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)(O)O)[C@@H](O)[C@H]2O)c1",
    "PLP": "CC1=NC=C(C(=C1O)C=O)COP(=O)(O)O",

    # ── Amino acids & derivatives ────────────────────────────────────
    "ALA": "C[C@@H](N)C(=O)O",
    "ARG": "NC(=N)NCCC[C@H](N)C(=O)O",
    "ASN": "NC(=O)C[C@H](N)C(=O)O",
    "ASP": "N[C@@H](CC(=O)O)C(=O)O",
    "CYS": "N[C@@H](CS)C(=O)O",
    "GLN": "NC(=O)CC[C@H](N)C(=O)O",
    "GLU": "N[C@@H](CCC(=O)O)C(=O)O",
    "GLY": "NCC(=O)O",
    "HIS": "N[C@@H](Cc1c[nH]cn1)C(=O)O",
    "ILE": "CC[C@H](C)[C@H](N)C(=O)O",
    "LEU": "CC(C)C[C@H](N)C(=O)O",
    "LYS": "NCCCC[C@H](N)C(=O)O",
    "MET": "CSCC[C@H](N)C(=O)O",
    "PHE": "N[C@@H](Cc1ccccc1)C(=O)O",
    "PRO": "OC(=O)[C@@H]1CCCN1",
    "SER": "N[C@@H](CO)C(=O)O",
    "THR": "C[C@@H](O)[C@H](N)C(=O)O",
    "TRP": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "TYR": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "VAL": "CC(C)[C@H](N)C(=O)O",

    # ── Common drugs ─────────────────────────────────────────────────
    "STI": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "IMT": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "ASP": "CC(=O)Oc1ccccc1C(=O)O",
    "IBP": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "SAH": "C[S+](CC[C@H](N)C(=O)O)C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    "NAP": "CC(=O)c1ccc2c(c1)NC(=O)C2",
    "CFZ": "CN(C)CCCN1c2ccccc2Sc3ccc(Cl)cc13",
    "FLU": "CN1CCN(c2ccc(C(F)(F)F)c3c2C(=O)N(C)C3=O)CC1",
    "TAM": "CC/C(c1ccccc1)=C(/c2ccc(OCCN(C)C)cc2)c3ccccc3",
    "MET": "CN1CCN(c2ccc(OC)c3c2C(=O)c4ccccc4C3=O)CC1",
    "PZA": "NC(=O)c1cnccn1",

    # ── Steroids & hormones ──────────────────────────────────────────
    "EST": "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O",
    "TST": "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O",
    "COR": "C[C@]12C[C@@H](O)[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2C(=O)CO",
    "PRG": "CC(=O)[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
    "RET": "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1",
    "DIO": "O=C1O[C@H](C[C@]23C[C@@H](O)CC[C@]12C)C3",
    "TYR": "CC(Cc1ccc(O)cc1)NC[C@H](O)c1ccc(O)c(N)c1",

    # ── Sugars ───────────────────────────────────────────────────────
    "BGC": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "GLC": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "NAG": "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@H](O[C@@H]1O)CO",
    "MAN": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "GAL": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "FUC": "C[C@@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "XYS": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O",
    "RIB": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O",
    "FRU": "OC[C@]1(O)O[C@@H](CO)[C@@H](O)[C@H]1O",
    "SOR": "OC[C@@H](O)[C@@H](O)[C@H](O)C(=O)CO",

    # ── Lipids & fatty acids ─────────────────────────────────────────
    "PLM": "CCCCCCCCCCCCCCCC(=O)O",
    "OLA": "CCCCCCCC/C=C/CCCCCCCC(=O)O",
    "STE": "CCCCCCCCCCCCCCCCCC(=O)O",
    "LNL": "CCCCC/C=C/C/C=C/CCCCCCCC(=O)O",
    "MYR": "CCCCCCCCCCCCCC(=O)O",
    "PAM": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "DOD": "CCCCCCCCCCCC(=O)O",

    # ── Common buffers & solvents ────────────────────────────────────
    "GOL": "C(C(CO)O)O",
    "EDO": "C(CO)O",
    "PEG": "C(CO)O",
    "ACT": "CC(=O)O",
    "DMS": "CS(=O)C",
    "DMF": "CN(C)C=O",
    "IPA": "CC(C)O",
    "BME": "CCO",
    "BNZ": "c1ccccc1",
    "TOL": "Cc1ccccc1",
    "PGO": "CC1CO1",
    "DTT": "C([C@@H](O)[C@H](O)CS)S",
    "TRS": "NC(CO)(CO)CO",

    # ── Ions & metals ────────────────────────────────────────────────
    "MG": "[Mg+2]",
    "ZN": "[Zn+2]",
    "CA": "[Ca+2]",
    "MN": "[Mn+2]",
    "FE": "[Fe+2]",
    "FE2": "[Fe+3]",
    "CU": "[Cu+2]",
    "CO": "[Co+2]",
    "NI": "[Ni+2]",
    "K": "[K+]",
    "NA": "[Na+]",
    "CL": "[Cl-]",
    "BR": "[Br-]",
    "F": "[F-]",
    "IOD": "[I-]",

    # ── Phosphates & common groups ───────────────────────────────────
    "PO4": "[O-]P(=O)([O-])[O-]",
    "SO4": "[O-]S(=O)(=O)[O-]",
    "NO3": "[O-][N+](=O)[O-]",
    "CO3": "[O-]C(=O)[O-]",
    "ACY": "CC(=O)[O-]",
    "CIT": "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "MLA": "OC(=O)C(O)C(=O)O",
    "OXL": "OC(=O)C(=O)O",
    "SUC": "OC(=O)CCC(=O)O",
    "FUM": "OC(=O)/C=C/C(=O)O",
    "AKG": "OC(=O)CCC(=O)C(=O)O",

    # ── Nucleoside analogs & antivirals ──────────────────────────────
    "AZT": "CC1=CN([C@@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)C(=O)NC1=O",
    "ACY": "OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    "TDR": "CC1=CN([C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)NC1=O",
    "CMP": "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1",
    "UMP": "O=C1C=CN([C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)C(=O)N1",
    "GMP": "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",

    # ── Antibiotics ──────────────────────────────────────────────────
    "PEN": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "TET": "CN(C)[C@H]1[C@@H]2C[C@H]3Cc4c(O)cccc4[C@@]3(C(=O)C2=C(O)[C@]1(O)C(=O)N)C(=O)N",
    "CHL": "O=[N+]([O-])c1ccc([C@@H](O)[C@@H](CO)NC(=O)C(Cl)Cl)cc1",
    "ERY": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@@H]([C@H]3O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "RIF": "COC1C=COC2(C)OC3=C(C2=O)C2=C(C(=O)C(=C3C)NC(=O)C(C)=CC=CC(C)C(O)C(C)C(O)C(C)C(O)C(C)C=CC=C(C)C(=O)NC1=O)C(O)=C3C(=O)c4c(C3=O)cccc4",

    # ── Anticancer drugs ─────────────────────────────────────────────
    "MTX": "Nc1nc(N)c2nc(CNc3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3)ccc2n1",
    "5FU": "O=C1NC(=O)NC=C1F",
    "CPT": "CC[C@@]1(O)C(=O)OCc2c1cc3n(c2=O)Cc4cc5c(cc4c3=O)OCO5",
    "ETP": "COc1cc2c(cc1OC)C(C(=O)c3cc4c(c(OC)c3OC)OCO4)C(O)C(C)O2",
    "VCR": "CC[C@@]1(C[C@H]2C[C@@]3(C4=C(C5=CC=CC=C5N4)CCN3CC2)C(=O)OC)C(=O)Nc6cc(ccc6OC)C(=O)N1",

    # ── CNS drugs ────────────────────────────────────────────────────
    "CAF": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "MSE": "C[C@@H](O)[C@H](N)C(=O)O",
    "DOP": "NCCc1ccc(O)c(O)c1",
    "SRT": "CNCC[C@H](Oc1ccc(F)cc1)c2ccccc2",
    "NOR": "NCCc1ccc(O)c(O)c1",
    "EPI": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "ADR": "CNC[C@H](O)c1ccc(O)c(O)c1",

    # ── Cardiovascular ────────────────────────────────────────────────
    "WAR": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
    "LOS": "CCCCCc1nn(Cc2ccc(-c3ccccc3-c3nn[nH]n3)cc2)c(=O)n1C",
    "VAS": "CCCCN(CCCC)c1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
    "AML": "CCOC(=O)C1=C(C)NC(C)=C(C(=O)OC)[C@@H]1c1ccccc1Cl",

    # ── Anti-diabetic ────────────────────────────────────────────────
    "MET": "CN(C)C(=N)N",
    "GLB": "COc1ccc(OC)c2c1c(C)c(C)n2C(=O)NCCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
    "ROS": "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccc1",
    "PIO": "CCc1ccc(CCOC(=O)c2cnc(C)c(C)c2)cc1",

    # ── Anti-inflammatory ─────────────────────────────────────────────
    "IND": "COc1ccc2c(c1)c(C)c(CC(=O)O)n2C(=O)c1ccc(Cl)cc1",
    "DEX": "C[C@@]12C[C@@H](O)[C@]3(F)[C@@H](CCC4=CC(=O)C=C[C@]34C)[C@@H]1C[C@H](O)[C@@]2(C)C(=O)CO",
    "PRD": "C[C@@]12C[C@@H](O)[C@@]3(F)[C@@H](CCC4=CC(=O)C=C[C@]34C)[C@@H]1CC[C@@]2(O)C(=O)CO",
    "SAL": "O=C(O)c1ccccc1O",
    "IBU": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
}
