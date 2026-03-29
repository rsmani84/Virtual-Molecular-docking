import gradio as gr
import pandas as pd
import requests
import os
from datetime import datetime
from rdkit.Chem import Descriptors, Lipinski, Crippen, AllChem
from Bio.PDB import PDBParser
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

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
# GLOBAL STATE
# -----------------------------
app_state = {
    "protein_pdb": None,
    "protein_name": "",
    "pdb_text": "",
    "protein_info": None,
    "ligand_mol": None,
    "ligand_smiles": "",
    "ligand_name": "",
    "ligand_info": None,
    "dock_box": {},
    "docking_result": None
}

# -----------------------------
# HELPERS
# -----------------------------
def fetch_pdb(pdb_id):
    try:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url, timeout=15)
        if response.status_code == 200 and "HEADER" in response.text[:500]:
            path = f"data/proteins/{pdb_id.upper()}.pdb"
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return path, response.text
        return None, None
    except:
        return None, None


def parse_pdb_info(pdb_path):
    try:
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
    except Exception as e:
        return {"Error": str(e)}


def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        return mol
    except:
        return None


def ligand_properties(mol):
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2)
    }


def simulate_docking(ligand_name, protein_name):
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
    summary_data = [[k, str(v)] for k, v in result.items() if k not in ["Interactions", "Interpretation"]]
    summary_table = Table(summary_data, colWidths=[220, 250])
    summary_table.setStyle(TableStyle([
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
# PROTEIN FUNCTIONS
# -----------------------------
def protein_fetch_fn(pdb_id):
    if not pdb_id.strip():
        return "Please enter a valid PDB ID.", pd.DataFrame(), "<p>No structure loaded.</p>"

    path, text = fetch_pdb(pdb_id.strip())
    if path:
        app_state["protein_pdb"] = path
        app_state["protein_name"] = pdb_id.upper()
        app_state["pdb_text"] = text
        info = parse_pdb_info(path)
        app_state["protein_info"] = info
        df = pd.DataFrame(info.items(), columns=["Property", "Value"])
        return f"Protein {pdb_id.upper()} fetched successfully.", df, f"<pre>{text[:3000]}</pre>"
    else:
        return "Unable to fetch PDB file. Please check the PDB ID.", pd.DataFrame(), "<p>No structure loaded.</p>"


def protein_upload_fn(file):
    if file is None:
        return "Please upload a PDB file.", pd.DataFrame(), "<p>No structure loaded.</p>"

    save_path = f"data/proteins/{os.path.basename(file.name)}"
    with open(file.name, "rb") as src:
        with open(save_path, "wb") as dst:
            dst.write(src.read())

    app_state["protein_pdb"] = save_path
    app_state["protein_name"] = os.path.basename(file.name)

    with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    app_state["pdb_text"] = text
    info = parse_pdb_info(save_path)
    app_state["protein_info"] = info
    df = pd.DataFrame(info.items(), columns=["Property", "Value"])

    return "Protein uploaded successfully.", df, f"<pre>{text[:3000]}</pre>"


# -----------------------------
# LIGAND FUNCTIONS
# -----------------------------
def ligand_prepare_fn(smiles, ligand_name):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return "Invalid SMILES. Please enter a valid ligand structure.", pd.DataFrame()

    app_state["ligand_mol"] = mol
    app_state["ligand_smiles"] = smiles
    app_state["ligand_name"] = ligand_name if ligand_name else "Ligand"

    props = ligand_properties(mol)
    app_state["ligand_info"] = props
    df = pd.DataFrame(props.items(), columns=["Property", "Value"])
    return "Ligand prepared successfully.", df


# -----------------------------
# DOCKING SETUP
# -----------------------------
def save_docking_box(center_x, center_y, center_z, size_x, size_y, size_z, exhaustiveness):
    app_state["dock_box"] = {
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "exhaustiveness": exhaustiveness
    }
    return "Docking box parameters saved.", str(app_state["dock_box"])


# -----------------------------
# RUN DOCKING
# -----------------------------
def run_docking_fn():
    if app_state["protein_pdb"] is None:
        return "Please prepare a protein first.", pd.DataFrame(), pd.DataFrame(), ""
    if app_state["ligand_mol"] is None:
        return "Please prepare a ligand first.", pd.DataFrame(), pd.DataFrame(), ""

    result = simulate_docking(app_state["ligand_name"], app_state["protein_name"])
    app_state["docking_result"] = result

    summary_df = pd.DataFrame([
        ["Protein", result["Protein"]],
        ["Ligand", result["Ligand"]],
        ["Binding Affinity (kcal/mol)", result["Binding Affinity (kcal/mol)"]],
        ["Pose Rank", result["Pose Rank"]],
        ["RMSD (Å)", result["RMSD (Å)"]]
    ], columns=["Parameter", "Value"])

    inter_df = pd.DataFrame(result["Interactions"], columns=["Interaction Type", "Residue", "Distance"])

    return "Docking completed successfully.", summary_df, inter_df, result["Interpretation"]


# -----------------------------
# PDF DOWNLOAD
# -----------------------------
def generate_report_fn(student_name, reg_no):
    result = app_state["docking_result"]
    if result is None:
        return None

    pdf_path = generate_pdf_report(
        student_name,
        reg_no,
        result,
        app_state.get("protein_info", None),
        app_state.get("ligand_info", None)
    )
    return pdf_path


# -----------------------------
# QUIZ
# -----------------------------
def evaluate_quiz(a1, a2, a3, a4):
    answers = [
        "Stronger predicted binding",
        "To avoid unwanted interference in docking",
        "Hydrogen bond",
        "Binding region for ligand"
    ]
    user_answers = [a1, a2, a3, a4]
    score = sum(1 for u, c in zip(user_answers, answers) if u == c)
    return f"Your Score: {score} / 4"


# -----------------------------
# FEEDBACK
# -----------------------------
def submit_feedback(name, reg_no, rating, feedback_text):
    save_feedback(name, reg_no, rating, feedback_text)

    df = pd.read_csv(FEEDBACK_FILE)
    avg_rating = round(df["rating"].mean(), 2) if not df.empty else 0
    total = len(df)

    return "Thank you for your feedback!", avg_rating, total


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Protein–Ligand Docking Virtual Lab") as demo:
    gr.Markdown("# 🧬 Protein–Ligand Docking Virtual Lab")
    gr.Markdown("An interactive educational platform to understand protein–ligand binding and molecular docking.")

    with gr.Tab("Home"):
        gr.Markdown("""
        ## Objective
        This virtual lab helps students learn the **workflow of molecular docking**:
        - Protein selection and preparation
        - Ligand input and molecular property analysis
        - Docking box setup
        - Docking score interpretation
        - Protein–ligand interaction analysis
        - Report generation and self-evaluation
        """)
        gr.Markdown("**Note:** This app is designed for teaching and learning purposes. The docking step uses an educational simulation.")

    with gr.Tab("Theory"):
        gr.Markdown("""
        ## 📘 Theory of Molecular Docking

        **Molecular docking** is a computational technique used to predict how a **ligand** binds to a **protein receptor**.

        ### Important Concepts
        - **Protein**: Biological target such as enzyme or receptor
        - **Ligand**: Small molecule/drug/phytochemical that binds to the protein
        - **Active Site**: Binding pocket where ligand interacts with the protein
        - **Docking Score**: Predicted binding energy (more negative = better affinity)
        - **Hydrogen Bond / Hydrophobic Interaction**: Important interactions stabilizing the complex
        - **RMSD**: Measure of structural similarity between poses
        """)

    with gr.Tab("Protein Preparation"):
        gr.Markdown("## 🧫 Protein Preparation")
        with gr.Row():
            pdb_id = gr.Textbox(label="Enter PDB ID", placeholder="Example: 6LU7")
            fetch_btn = gr.Button("Fetch Protein")

        protein_file = gr.File(label="Or Upload PDB File", file_types=[".pdb"])
        upload_btn = gr.Button("Upload Protein")

        protein_status = gr.Textbox(label="Status")
        protein_df = gr.Dataframe(label="Protein Summary")
        protein_view = gr.HTML(label="Protein Structure Preview")

        fetch_btn.click(protein_fetch_fn, inputs=pdb_id, outputs=[protein_status, protein_df, protein_view])
        upload_btn.click(protein_upload_fn, inputs=protein_file, outputs=[protein_status, protein_df, protein_view])

    with gr.Tab("Ligand Preparation"):
        gr.Markdown("## 💊 Ligand Preparation")
        smiles = gr.Textbox(label="Enter Ligand SMILES", placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O")
        ligand_name = gr.Textbox(label="Ligand Name (Optional)", placeholder="Example: Aspirin")
        ligand_btn = gr.Button("Prepare Ligand")

        ligand_status = gr.Textbox(label="Status")
        ligand_df = gr.Dataframe(label="Ligand Properties")

        ligand_btn.click(ligand_prepare_fn, inputs=[smiles, ligand_name], outputs=[ligand_status, ligand_df])

    with gr.Tab("Docking Setup"):
        gr.Markdown("## 🎯 Docking Setup")
        with gr.Row():
            center_x = gr.Number(label="center_x", value=10.0)
            center_y = gr.Number(label="center_y", value=15.0)
            center_z = gr.Number(label="center_z", value=20.0)

        with gr.Row():
            size_x = gr.Number(label="size_x", value=20.0)
            size_y = gr.Number(label="size_y", value=20.0)
            size_z = gr.Number(label="size_z", value=20.0)

        exhaustiveness = gr.Slider(1, 20, value=8, step=1, label="Exhaustiveness")
        dock_btn = gr.Button("Save Docking Box")

        dock_status = gr.Textbox(label="Status")
        dock_json = gr.Textbox(label="Saved Parameters")

        dock_btn.click(
            save_docking_box,
            inputs=[center_x, center_y, center_z, size_x, size_y, size_z, exhaustiveness],
            outputs=[dock_status, dock_json]
        )

    with gr.Tab("Run Docking"):
        gr.Markdown("## ⚙️ Run Docking")
        gr.Markdown("**Educational Mode:** This version simulates docking results for teaching and learning.")
        run_btn = gr.Button("Run Demo Docking")

        run_status = gr.Textbox(label="Status")
        result_summary = gr.Dataframe(label="Docking Summary")
        result_interactions = gr.Dataframe(label="Protein–Ligand Interactions")
        result_interpretation = gr.Textbox(label="Interpretation")

        run_btn.click(run_docking_fn, outputs=[run_status, result_summary, result_interactions, result_interpretation])

    with gr.Tab("Results / Report"):
        gr.Markdown("## 📊 Generate Report")
        student_name = gr.Textbox(label="Student Name (Optional)")
        reg_no = gr.Textbox(label="Registration Number (Optional)")
        report_btn = gr.Button("Generate PDF Report")
        report_file = gr.File(label="Download Report")

        report_btn.click(generate_report_fn, inputs=[student_name, reg_no], outputs=report_file)

    with gr.Tab("Quiz / Viva"):
        gr.Markdown("## 📝 Quiz / Viva")
        q1 = gr.Radio(
            ["Weaker binding", "Stronger predicted binding", "Protein denaturation", "Ligand instability"],
            label="1. What does a more negative docking score indicate?"
        )
        q2 = gr.Radio(
            ["To reduce file size only", "Because they are always harmful", "To avoid unwanted interference in docking", "To increase ligand size"],
            label="2. Why are water molecules often removed during protein preparation?"
        )
        q3 = gr.Radio(
            ["Hydrogen bond", "Gamma decay", "Radioactivity", "Nuclear fission"],
            label="3. Which interaction is important for ligand binding?"
        )
        q4 = gr.Radio(
            ["Protein synthesis", "Binding region for ligand", "DNA replication", "Membrane transport"],
            label="4. What is the role of the active site?"
        )
        quiz_btn = gr.Button("Submit Quiz")
        quiz_result = gr.Textbox(label="Quiz Result")

        quiz_btn.click(evaluate_quiz, inputs=[q1, q2, q3, q4], outputs=quiz_result)

    with gr.Tab("Feedback"):
        gr.Markdown("## ⭐ Feedback")
        fb_name = gr.Textbox(label="Name (Optional)")
        fb_reg = gr.Textbox(label="Registration Number (Optional)")
        fb_rating = gr.Slider(1, 5, value=4, step=1, label="Rate this virtual lab")
        fb_text = gr.Textbox(label="Write your feedback", lines=4)
        fb_btn = gr.Button("Submit Feedback")

        fb_status = gr.Textbox(label="Status")
        avg_rating = gr.Number(label="Average Rating")
        total_feedback = gr.Number(label="Total Feedback Entries")

        fb_btn.click(submit_feedback, inputs=[fb_name, fb_reg, fb_rating, fb_text], outputs=[fb_status, avg_rating, total_feedback])

    gr.Markdown("---")
    gr.Markdown("Developed using **Gradio** for educational demonstration of protein–ligand docking workflow.")

demo.launch()
