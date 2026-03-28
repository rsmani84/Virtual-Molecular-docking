# Protein–Ligand Docking Virtual Lab (Streamlit)
# Save this as app.py

import streamlit as st
import pandas as pd
import requests
import os
import io
import base64
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, AllChem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import py3Dmol
from streamlit.components.v1 import html
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Protein–Ligand Docking Virtual Lab",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# FOLDERS
# -----------------------------
os.makedirs("data/proteins", exist_ok=True)
os.makedirs("data/ligands", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

FEEDBACK_FILE = "feedback/feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["timestamp", "name", "reg_no", "rating", "feedback"]).to_csv(FEEDBACK_FILE, index=False)

# -----------------------------
# SESSION STATE
# -----------------------------
def init_session():
    defaults = {
        "protein_pdb": None,
        "protein_name": "",
        "ligand_mol": None,
        "ligand_smiles": "",
        "ligand_name": "",
        "docking_result": None,
        "student_name": "",
        "reg_no": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
}
.sub-title {
    font-size: 1.1rem;
    color: #475569;
}
.card {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.small-note {
    font-size: 0.9rem;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
def fetch_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = requests.get(url, timeout=15)
    if response.status_code == 200 and "HEADER" in response.text[:200]:
        path = f"data/proteins/{pdb_id.upper()}.pdb"
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
        return path, response.text
    return None, None


def parse_pdb_info(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    chains = list(structure.get_chains())
    residues = list(structure.get_residues())
    waters = sum(1 for r in residues if r.get_resname() == "HOH")
    hetero = sum(1 for r in residues if r.id[0] != " ")
    return {
        "Chains": len(chains),
        "Residues": len(residues),
        "Water molecules": waters,
        "Hetero residues": hetero,
    }


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def ligand_properties(mol):
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2)
    }



def show_3dmol(pdb_text=None, mol_block=None, height=450):
    view = py3Dmol.view(width=800, height=height)
    if pdb_text:
        view.addModel(pdb_text, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    if mol_block:
        view.addModel(mol_block, 'mol')
        view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
    view.zoomTo()
    return view._make_html()


def simulate_docking(ligand_name, protein_name):
    # Educational simulation only
    base_score = -6.2
    if ligand_name:
        base_score -= min(len(ligand_name) * 0.05, 1.0)
    residues = ["ASN142", "GLU166", "HIS41", "SER144", "CYS145"]
    interactions = [
        ["Hydrogen Bond", residues[0], "2.9 Å"],
        ["Hydrogen Bond", residues[1], "3.1 Å"],
        ["Hydrophobic", residues[2], "4.5 Å"],
        ["Van der Waals", residues[3], "3.8 Å"],
        ["Hydrophobic", residues[4], "4.2 Å"],
    ]
    return {
        "Protein": protein_name if protein_name else "Uploaded Protein",
        "Ligand": ligand_name if ligand_name else "Uploaded Ligand",
        "Binding Affinity (kcal/mol)": round(base_score, 2),
        "Pose Rank": 1,
        "RMSD (Å)": 1.78,
        "Interactions": interactions,
        "Interpretation": "A more negative docking score indicates stronger predicted binding affinity. This result suggests moderate to good ligand–protein interaction in the selected binding pocket."
    }


def generate_pdf_report(student_name, reg_no, result, protein_info=None, ligand_info=None):
    file_path = "docking_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Protein–Ligand Docking Virtual Lab Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"Name: {student_name if student_name else 'Optional / Not Provided'}", styles['Normal']))
    story.append(Paragraph(f"Registration Number: {reg_no if reg_no else 'Optional / Not Provided'}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Docking Summary", styles['Heading2']))
    summary_data = [[k, str(v)] for k, v in result.items() if k != "Interactions" and k != "Interpretation"]
    summary_table = Table(summary_data, colWidths=[220, 250])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    if protein_info:
        story.append(Paragraph("Protein Information", styles['Heading2']))
        protein_table = Table([[k, str(v)] for k, v in protein_info.items()], colWidths=[220, 250])
        protein_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(protein_table)
        story.append(Spacer(1, 12))

    if ligand_info:
        story.append(Paragraph("Ligand Properties", styles['Heading2']))
        ligand_table = Table([[k, str(v)] for k, v in ligand_info.items()], colWidths=[220, 250])
        ligand_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(ligand_table)
        story.append(Spacer(1, 12))

    story.append(Paragraph("Protein–Ligand Interactions", styles['Heading2']))
    interaction_data = [["Interaction Type", "Residue", "Distance"]] + result["Interactions"]
    interaction_table = Table(interaction_data, colWidths=[160, 160, 150])
    interaction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(interaction_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Interpretation", styles['Heading2']))
    story.append(Paragraph(result["Interpretation"], styles['BodyText']))

    doc.build(story)
    return file_path


def save_feedback(name, reg_no, rating, feedback_text):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "reg_no": reg_no,
        "rating": rating,
        "feedback": feedback_text
    }])
    row.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)

# -----------------------------
# SIDEBAR
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Theory", "Protein Preparation", "Ligand Preparation", "Docking Setup", "Run Docking", "Results", "Quiz / Viva", "Feedback"]
)

st.sidebar.markdown("---")
st.sidebar.text_input("Student Name (Optional)", key="student_name")
st.sidebar.text_input("Registration Number (Optional)", key="reg_no")

# -----------------------------
# HOME
# -----------------------------
if menu == "Home":
    st.markdown('<div class="main-title">🧬 Protein–Ligand Docking Virtual Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An interactive educational platform to understand protein–ligand binding and molecular docking.</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Objective
        This virtual lab helps students learn the **workflow of molecular docking**:
        - Protein selection and preparation
        - Ligand input and molecular property analysis
        - Docking box setup
        - Docking score interpretation
        - Protein–ligand interaction analysis
        - Report generation and self-evaluation
        """)

        st.info("This app is designed for **teaching and learning purposes**. The docking step uses an educational simulation by default so it can run on Streamlit Cloud without heavy dependencies.")

    with col2:
        st.success("Recommended workflow:\n1. Protein Preparation\n2. Ligand Preparation\n3. Docking Setup\n4. Run Docking\n5. Results")

# -----------------------------
# THEORY
# -----------------------------
elif menu == "Theory":
    st.header("📘 Theory of Molecular Docking")
    st.markdown("""
    **Molecular docking** is a computational technique used to predict how a **ligand** binds to a **protein receptor**.

    ### Important Concepts
    - **Protein**: Biological target such as enzyme or receptor
    - **Ligand**: Small molecule/drug/phytochemical that binds to the protein
    - **Active Site**: Binding pocket where ligand interacts with the protein
    - **Docking Score**: Predicted binding energy (more negative = better affinity)
    - **Hydrogen Bond / Hydrophobic Interaction**: Important interactions stabilizing the complex
    - **RMSD**: Measure of structural similarity between poses

    ### Applications
    - Drug discovery
    - Drug repurposing
    - Enzyme inhibition studies
    - Natural product screening
    - Protein–ligand interaction analysis
    """)

# -----------------------------
# PROTEIN PREPARATION
# -----------------------------
elif menu == "Protein Preparation":
    st.header("🧫 Protein Preparation")
    method = st.radio("Choose Protein Input Method", ["Fetch by PDB ID", "Upload PDB File"])

    protein_info = None
    pdb_text = None

    if method == "Fetch by PDB ID":
        pdb_id = st.text_input("Enter PDB ID", placeholder="Example: 6LU7")
        if st.button("Fetch Protein"):
            if pdb_id.strip():
                path, text = fetch_pdb(pdb_id.strip())
                if path:
                    st.session_state.protein_pdb = path
                    st.session_state.protein_name = pdb_id.upper()
                    st.success(f"Protein {pdb_id.upper()} fetched successfully.")
                else:
                    st.error("Unable to fetch PDB file. Please check the PDB ID.")

    else:
        uploaded = st.file_uploader("Upload Protein PDB File", type=["pdb"])
        if uploaded is not None:
            save_path = f"data/proteins/{uploaded.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded.read())
            st.session_state.protein_pdb = save_path
            st.session_state.protein_name = uploaded.name
            st.success("Protein uploaded successfully.")

    if st.session_state.protein_pdb:
        with open(st.session_state.protein_pdb, "r", encoding="utf-8", errors="ignore") as f:
            pdb_text = f.read()
        protein_info = parse_pdb_info(st.session_state.protein_pdb)
        st.subheader("Protein Summary")
        st.dataframe(pd.DataFrame(protein_info.items(), columns=["Property", "Value"]), use_container_width=True)
        st.subheader("3D Protein View")
        html(show_3dmol(pdb_text=pdb_text), height=500)
        st.session_state["protein_info"] = protein_info
        st.session_state["pdb_text"] = pdb_text

# -----------------------------
# LIGAND PREPARATION
# -----------------------------
elif menu == "Ligand Preparation":
    st.header("💊 Ligand Preparation")
    smiles = st.text_input("Enter Ligand SMILES", placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O")
    ligand_name = st.text_input("Ligand Name (Optional)", placeholder="Example: Aspirin")

    if st.button("Prepare Ligand"):
        mol = smiles_to_mol(smiles)
        if mol:
            st.session_state.ligand_mol = mol
            st.session_state.ligand_smiles = smiles
            st.session_state.ligand_name = ligand_name if ligand_name else "Ligand"
            st.success("Ligand prepared successfully.")
        else:
            st.error("Invalid SMILES. Please enter a valid ligand structure.")

    if st.session_state.ligand_mol is not None:
        mol = st.session_state.ligand_mol
        props = ligand_properties(mol)
        st.subheader("Ligand Properties")
        st.dataframe(pd.DataFrame(props.items(), columns=["Property", "Value"]), use_container_width=True)
        st.info("Ligand prepared successfully. 2D structure preview is disabled in this deployed version.")
        st.session_state["ligand_info"] = props

# -----------------------------
# DOCKING SETUP
# -----------------------------
elif menu == "Docking Setup":
    st.header("🎯 Docking Setup")
    st.markdown("Define the docking search space (binding pocket / grid box).")

    col1, col2, col3 = st.columns(3)
    with col1:
        center_x = st.number_input("center_x", value=10.0)
        center_y = st.number_input("center_y", value=15.0)
    with col2:
        center_z = st.number_input("center_z", value=20.0)
        size_x = st.number_input("size_x", value=20.0)
    with col3:
        size_y = st.number_input("size_y", value=20.0)
        size_z = st.number_input("size_z", value=20.0)

    exhaustiveness = st.slider("Exhaustiveness", 1, 20, 8)

    st.session_state["dock_box"] = {
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "exhaustiveness": exhaustiveness
    }

    st.success("Docking box parameters saved.")
    st.json(st.session_state["dock_box"])

# -----------------------------
# RUN DOCKING
# -----------------------------
elif menu == "Run Docking":
    st.header("⚙️ Run Docking")
    st.warning("Educational Mode: This version simulates docking results for teaching and learning.")

    if st.session_state.protein_pdb is None:
        st.error("Please prepare a protein first.")
    elif st.session_state.ligand_mol is None:
        st.error("Please prepare a ligand first.")
    else:
        if st.button("Run Demo Docking"):
            result = simulate_docking(st.session_state.ligand_name, st.session_state.protein_name)
            st.session_state.docking_result = result
            st.success("Docking completed successfully.")
            st.json({k: v for k, v in result.items() if k != "Interactions"})

# -----------------------------
# RESULTS
# -----------------------------
elif menu == "Results":
    st.header("📊 Docking Results")
    result = st.session_state.docking_result

    if result is None:
        st.info("No docking results available yet. Please run docking first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Binding Affinity", f"{result['Binding Affinity (kcal/mol)']} kcal/mol")
        c2.metric("Pose Rank", result["Pose Rank"])
        c3.metric("RMSD", f"{result['RMSD (Å)']} Å")

        st.subheader("Docking Summary")
        summary_df = pd.DataFrame([
            ["Protein", result["Protein"]],
            ["Ligand", result["Ligand"]],
            ["Binding Affinity (kcal/mol)", result["Binding Affinity (kcal/mol)"]],
            ["Pose Rank", result["Pose Rank"]],
            ["RMSD (Å)", result["RMSD (Å)"]]
        ], columns=["Parameter", "Value"])
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Protein–Ligand Interactions")
        inter_df = pd.DataFrame(result["Interactions"], columns=["Interaction Type", "Residue", "Distance"])
        st.dataframe(inter_df, use_container_width=True)

        st.subheader("Interpretation")
        st.success(result["Interpretation"])

        pdf_path = generate_pdf_report(
            st.session_state.student_name,
            st.session_state.reg_no,
            result,
            st.session_state.get("protein_info", None),
            st.session_state.get("ligand_info", None)
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Docking Report (PDF)",
                f,
                file_name="docking_report.pdf",
                mime="application/pdf"
            )

# -----------------------------
# QUIZ / VIVA
# -----------------------------
elif menu == "Quiz / Viva":
    st.header("📝 Quiz / Viva")

    questions = {
        "1. What does a more negative docking score indicate?": [
            "Weaker binding",
            "Stronger predicted binding",
            "Protein denaturation",
            "Ligand instability"
        ],
        "2. Why are water molecules often removed during protein preparation?": [
            "To reduce file size only",
            "Because they are always harmful",
            "To avoid unwanted interference in docking",
            "To increase ligand size"
        ],
        "3. Which interaction is important for ligand binding?": [
            "Hydrogen bond",
            "Gamma decay",
            "Radioactivity",
            "Nuclear fission"
        ],
        "4. What is the role of the active site?": [
            "Protein synthesis",
            "Binding region for ligand",
            "DNA replication",
            "Membrane transport"
        ]
    }

    answers = {
        0: "Stronger predicted binding",
        1: "To avoid unwanted interference in docking",
        2: "Hydrogen bond",
        3: "Binding region for ligand"
    }

    score = 0
    user_answers = []
    for i, (q, opts) in enumerate(questions.items()):
        ans = st.radio(q, opts, key=f"q_{i}")
        user_answers.append(ans)

    if st.button("Submit Quiz"):
        for i, ans in enumerate(user_answers):
            if ans == answers[i]:
                score += 1
        st.success(f"Your Score: {score} / {len(questions)}")

# -----------------------------
# FEEDBACK
# -----------------------------
elif menu == "Feedback":
    st.header("⭐ Feedback")
    st.write("Your feedback helps improve this virtual lab.")

    name = st.text_input("Name (Optional)")
    reg_no = st.text_input("Registration Number (Optional)")
    rating = st.slider("Rate this virtual lab", 1, 5, 4)
    feedback_text = st.text_area("Write your feedback")

    if st.button("Submit Feedback"):
        save_feedback(name, reg_no, rating, feedback_text)
        st.success("Thank you for your feedback!")

    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        if not df.empty:
            st.subheader("Overall Feedback Summary")
            st.metric("Average Rating", round(df["rating"].mean(), 2))
            st.metric("Total Feedback Entries", len(df))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit for educational demonstration of protein–ligand docking workflow.")

# requirements.txt
# streamlit
# pandas
# requests
# rdkit
# biopython
# py3Dmol
# reportlab

# README.md (quick setup)
# 1. pip install -r requirements.txt
# 2. streamlit run app.py
# 3. Upload to GitHub
# 4. Deploy on Streamlit Cloud
