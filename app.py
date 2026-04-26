import streamlit as st
import numpy as np
import json, os, time
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="CropGuard AI", page_icon="🌿", layout="wide")

# ── Disease database ──────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab":                               {"crop":"Apple",      "disease":"Apple Scab",                  "sev":"mild",    "area":"15–35%", "urgency":"Treat within 1 week",  "season":"Cool & wet spring"},
    "Apple___Black_rot":                                {"crop":"Apple",      "disease":"Black Rot",                   "sev":"severe",  "area":"40–70%", "urgency":"Treat immediately",     "season":"Warm humid summers"},
    "Apple___Cedar_apple_rust":                         {"crop":"Apple",      "disease":"Cedar Apple Rust",            "sev":"mild",    "area":"10–30%", "urgency":"Monitor & treat",       "season":"Spring months"},
    "Apple___healthy":                                  {"crop":"Apple",      "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Blueberry___healthy":                              {"crop":"Blueberry",  "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Cherry_(including_sour)__Powdery_mildew":          {"crop":"Cherry",     "disease":"Powdery Mildew",              "sev":"mild",    "area":"20–40%", "urgency":"Treat within 1 week",  "season":"Dry warm periods"},
    "Cherry(including_sour)__healthy":                  {"crop":"Cherry",     "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot": {"crop":"Corn",       "disease":"Cercospora / Gray Leaf Spot", "sev":"mild",    "area":"15–35%", "urgency":"Monitor & treat",       "season":"Warm & humid months"},
    "Corn(maize)_Common_rust":                          {"crop":"Corn",       "disease":"Common Rust",                 "sev":"mild",    "area":"20–40%", "urgency":"Treat within 1 week",  "season":"Summer months"},
    "Corn(maize)__Northern_Leaf_Blight":                {"crop":"Corn",       "disease":"Northern Leaf Blight",        "sev":"severe",  "area":"40–60%", "urgency":"Treat immediately",     "season":"Cool & wet periods"},
    "Corn(maize)healthy":                               {"crop":"Corn",       "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Grape___Black_rot":                                {"crop":"Grape",      "disease":"Black Rot",                   "sev":"severe",  "area":"50–80%", "urgency":"Treat immediately",     "season":"Warm & humid summers"},
    "Grape___Esca(Black_Measles)":                      {"crop":"Grape",      "disease":"Esca (Black Measles)",        "sev":"severe",  "area":"40–70%", "urgency":"Treat immediately",     "season":"Hot dry summers"},
    "Grape___Leaf_blight(Isariopsis_Leaf_Spot)":        {"crop":"Grape",      "disease":"Leaf Blight",                 "sev":"mild",    "area":"20–45%", "urgency":"Treat within 1 week",  "season":"Late summer"},
    "Grape___healthy":                                  {"crop":"Grape",      "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Orange___Haunglongbing(Citrus_greening)":          {"crop":"Orange",     "disease":"Citrus Greening (HLB)",       "sev":"severe",  "area":"60–90%", "urgency":"Treat immediately",     "season":"Year-round threat"},
    "Peach___Bacterial_spot":                           {"crop":"Peach",      "disease":"Bacterial Spot",              "sev":"mild",    "area":"15–40%", "urgency":"Treat within 1 week",  "season":"Warm & wet periods"},
    "Peach___healthy":                                  {"crop":"Peach",      "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Pepper,_bell___Bacterial_spot":                    {"crop":"Pepper",     "disease":"Bacterial Spot",              "sev":"mild",    "area":"15–35%", "urgency":"Monitor & treat",       "season":"Warm humid months"},
    "Pepper,_bell___healthy":                           {"crop":"Pepper",     "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Potato___Early_blight":                            {"crop":"Potato",     "disease":"Early Blight",                "sev":"mild",    "area":"20–40%", "urgency":"Treat within 1 week",  "season":"Warm & humid months"},
    "Potato___Late_blight":                             {"crop":"Potato",     "disease":"Late Blight",                 "sev":"severe",  "area":"50–80%", "urgency":"Treat immediately",     "season":"Cool & wet conditions"},
    "Potato___healthy":                                 {"crop":"Potato",     "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Raspberry___healthy":                              {"crop":"Raspberry",  "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Soybean___healthy":                                {"crop":"Soybean",    "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Squash___Powdery_mildew":                          {"crop":"Squash",     "disease":"Powdery Mildew",              "sev":"mild",    "area":"20–45%", "urgency":"Treat within 1 week",  "season":"Dry warm periods"},
    "Strawberry___Leaf_scorch":                         {"crop":"Strawberry", "disease":"Leaf Scorch",                 "sev":"mild",    "area":"15–35%", "urgency":"Monitor & treat",       "season":"Hot dry summers"},
    "Strawberry___healthy":                             {"crop":"Strawberry", "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
    "Tomato___Bacterial_spot":                          {"crop":"Tomato",     "disease":"Bacterial Spot",              "sev":"mild",    "area":"15–35%", "urgency":"Treat within 1 week",  "season":"Warm & wet periods"},
    "Tomato___Early_blight":                            {"crop":"Tomato",     "disease":"Early Blight",                "sev":"mild",    "area":"20–40%", "urgency":"Treat within 1 week",  "season":"Warm & humid months"},
    "Tomato___Late_blight":                             {"crop":"Tomato",     "disease":"Late Blight",                 "sev":"severe",  "area":"50–80%", "urgency":"Treat immediately",     "season":"Cool & wet conditions"},
    "Tomato___Leaf_Mold":                               {"crop":"Tomato",     "disease":"Leaf Mold",                   "sev":"mild",    "area":"20–45%", "urgency":"Treat within 1 week",  "season":"High humidity periods"},
    "Tomato___Septoria_leaf_spot":                      {"crop":"Tomato",     "disease":"Septoria Leaf Spot",          "sev":"mild",    "area":"15–40%", "urgency":"Treat within 1 week",  "season":"Wet & warm months"},
    "Tomato___Spider_mites Two-spotted_spider_mite":    {"crop":"Tomato",     "disease":"Spider Mites",                "sev":"mild",    "area":"10–30%", "urgency":"Monitor & treat",       "season":"Hot dry summers"},
    "Tomato___Target_Spot":                             {"crop":"Tomato",     "disease":"Target Spot",                 "sev":"mild",    "area":"20–40%", "urgency":"Treat within 1 week",  "season":"Warm & humid months"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":           {"crop":"Tomato",     "disease":"Yellow Leaf Curl Virus",      "sev":"severe",  "area":"60–90%", "urgency":"Treat immediately",     "season":"Year-round threat"},
    "Tomato___Tomato_mosaic_virus":                     {"crop":"Tomato",     "disease":"Mosaic Virus",                "sev":"severe",  "area":"50–80%", "urgency":"Treat immediately",     "season":"Year-round threat"},
    "Tomato___healthy":                                 {"crop":"Tomato",     "disease":"Healthy",                     "sev":"healthy", "area":"0%",     "urgency":"No action needed",      "season":"All seasons"},
}

# ── Treatment database ────────────────────────────────────────
TREATMENT_INFO = {
    "Apple___Apple_scab": {
        "chemical": "Apply fungicides like Captan or Mancozeb every 7–10 days during wet seasons.",
        "organic":  "Use sulfur-based sprays or neem oil. Remove and destroy infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation by pruning."
    },
    "Apple___Black_rot": {
        "chemical": "Spray Captan or Thiophanate-methyl fungicide at bud break and repeat every 10 days.",
        "organic":  "Prune out infected branches. Apply copper-based fungicides.",
        "prevention": "Remove mummified fruits. Avoid overhead irrigation."
    },
    "Apple___Cedar_apple_rust": {
        "chemical": "Apply Myclobutanil or Propiconazole fungicide from pink bud stage through petal fall.",
        "organic":  "Use sulfur sprays during early spring. Remove nearby juniper/cedar trees.",
        "prevention": "Plant rust-resistant apple varieties. Maintain distance from cedar trees."
    },
    "Apple___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Continue regular maintenance and monitoring.",
        "prevention": "Maintain proper watering, fertilization, and annual pruning."
    },
    "Blueberry___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Keep soil acidic (pH 4.5–5.5) with pine bark mulch.",
        "prevention": "Monitor for pests regularly. Ensure well-draining soil."
    },
    "Cherry_(including_sour)__Powdery_mildew": {
        "chemical": "Apply Myclobutanil or Trifloxystrobin fungicide at first sign of symptoms.",
        "organic":  "Spray diluted baking soda solution (1 tbsp per litre). Use neem oil.",
        "prevention": "Avoid overhead watering. Prune for airflow. Avoid excess nitrogen."
    },
    "Cherry(including_sour)__healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Regular compost and balanced fertilization.",
        "prevention": "Annual pruning and pest scouting."
    },
    "Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot": {
        "chemical": "Apply Azoxystrobin or Pyraclostrobin fungicide at tasseling stage.",
        "organic":  "Rotate crops with non-host plants. Use resistant hybrids.",
        "prevention": "Avoid planting corn continuously. Till crop residue after harvest."
    },
    "Corn(maize)_Common_rust": {
        "chemical": "Apply Propiconazole or Triazole fungicide at early rust detection.",
        "organic":  "Plant resistant hybrid varieties. Early planting reduces risk.",
        "prevention": "Scout fields weekly. Plant early-maturing resistant varieties."
    },
    "Corn(maize)__Northern_Leaf_Blight": {
        "chemical": "Apply Azoxystrobin or Mancozeb fungicide from V8 stage onwards.",
        "organic":  "Use resistant hybrids. Crop rotation with soybean or wheat.",
        "prevention": "Avoid dense plant populations. Remove crop debris post-harvest."
    },
    "Corn(maize)healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Balanced NPK fertilization and regular irrigation.",
        "prevention": "Scout for pests and diseases weekly."
    },
    "Grape___Black_rot": {
        "chemical": "Apply Mancozeb or Myclobutanil from bud break. Repeat every 7–10 days.",
        "organic":  "Remove and destroy all infected berries and leaves. Use copper sprays.",
        "prevention": "Prune for good air circulation. Avoid overhead irrigation."
    },
    "Grape___Esca(Black_Measles)": {
        "chemical": "No curative fungicide. Prune infected wood and apply wound sealant.",
        "organic":  "Remove severely infected vines. Apply Trichoderma-based biocontrol.",
        "prevention": "Avoid large pruning wounds. Sanitize pruning tools between plants."
    },
    "Grape___Leaf_blight(Isariopsis_Leaf_Spot)": {
        "chemical": "Apply Mancozeb or Copper oxychloride fungicide at 10-day intervals.",
        "organic":  "Remove infected leaves. Improve canopy ventilation.",
        "prevention": "Avoid leaf wetness. Thin canopy to improve air circulation."
    },
    "Grape___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Apply balanced fertilizer and maintain proper trellising.",
        "prevention": "Regular pruning and canopy management."
    },
    "Orange___Haunglongbing(Citrus_greening)": {
        "chemical": "No cure exists. Remove and destroy infected trees immediately.",
        "organic":  "Control Asian citrus psyllid vector with neem oil or insecticidal soap.",
        "prevention": "Use certified disease-free planting material. Install psyllid barriers."
    },
    "Peach___Bacterial_spot": {
        "chemical": "Apply copper-based bactericides (Kocide) weekly from shuck split to harvest.",
        "organic":  "Use copper hydroxide sprays. Avoid overhead irrigation.",
        "prevention": "Plant resistant varieties. Avoid wounding fruit during thinning."
    },
    "Peach___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Balanced fertilization and proper thinning of fruits.",
        "prevention": "Annual pruning and dormant copper sprays as preventive."
    },
    "Pepper,_bell___Bacterial_spot": {
        "chemical": "Apply copper bactericide + Mancozeb mixture every 5–7 days in wet weather.",
        "organic":  "Use copper hydroxide sprays. Remove infected plant material.",
        "prevention": "Use disease-free certified seeds. Avoid overhead irrigation."
    },
    "Pepper,_bell___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Regular compost and proper spacing for air circulation.",
        "prevention": "Rotate crops. Avoid waterlogged soils."
    },
    "Potato___Early_blight": {
        "chemical": "Apply Chlorothalonil or Mancozeb every 7–10 days starting at first sign.",
        "organic":  "Use copper-based fungicides. Remove and destroy lower infected leaves.",
        "prevention": "Ensure adequate plant nutrition, especially nitrogen. Avoid overhead watering."
    },
    "Potato___Late_blight": {
        "chemical": "Apply Metalaxyl or Cymoxanil fungicide immediately. Repeat every 5–7 days.",
        "organic":  "Use copper hydroxide sprays. Destroy all infected plant material immediately.",
        "prevention": "Plant certified disease-free seed potatoes. Avoid excessive moisture."
    },
    "Potato___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Hill up plants and maintain soil moisture evenly.",
        "prevention": "Use certified seed. Rotate crops every 3–4 years."
    },
    "Raspberry___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Mulch around plants to retain moisture and suppress weeds.",
        "prevention": "Prune out old canes after fruiting. Ensure good drainage."
    },
    "Soybean___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Inoculate seeds with Rhizobium before planting.",
        "prevention": "Rotate with non-legume crops. Monitor for aphids and caterpillars."
    },
    "Squash___Powdery_mildew": {
        "chemical": "Apply Sulfur or Myclobutanil fungicide at first sign of white powder.",
        "organic":  "Spray diluted milk solution (1:9 ratio). Use potassium bicarbonate.",
        "prevention": "Plant in full sun. Avoid overhead watering. Space plants widely."
    },
    "Strawberry___Leaf_scorch": {
        "chemical": "Apply Captan or Myclobutanil fungicide every 10–14 days.",
        "organic":  "Remove and destroy infected leaves. Use copper-based sprays.",
        "prevention": "Avoid overhead irrigation. Renovate beds after harvest."
    },
    "Strawberry___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Apply balanced fertilizer after renovation.",
        "prevention": "Renovate beds annually. Use drip irrigation."
    },
    "Tomato___Bacterial_spot": {
        "chemical": "Apply copper bactericide + Mancozeb every 5–7 days during wet periods.",
        "organic":  "Use copper hydroxide sprays. Remove infected leaves immediately.",
        "prevention": "Use disease-free seeds. Avoid working with plants when wet."
    },
    "Tomato___Early_blight": {
        "chemical": "Apply Chlorothalonil or Mancozeb every 7 days from first symptoms.",
        "organic":  "Remove lower infected leaves. Spray neem oil or copper fungicide.",
        "prevention": "Mulch soil to prevent spore splash. Rotate crops annually."
    },
    "Tomato___Late_blight": {
        "chemical": "Apply Metalaxyl or Cymoxanil fungicide immediately. Repeat every 5 days.",
        "organic":  "Spray copper hydroxide. Destroy all infected plant material urgently.",
        "prevention": "Avoid overhead watering. Ensure good air circulation between plants."
    },
    "Tomato___Leaf_Mold": {
        "chemical": "Apply Chlorothalonil or Copper fungicide at 7-day intervals.",
        "organic":  "Improve greenhouse ventilation. Remove infected leaves immediately.",
        "prevention": "Keep humidity below 85%. Avoid wetting foliage when irrigating."
    },
    "Tomato___Septoria_leaf_spot": {
        "chemical": "Apply Mancozeb or Chlorothalonil every 7–10 days.",
        "organic":  "Remove infected lower leaves. Apply copper-based fungicide spray.",
        "prevention": "Mulch around plants. Avoid overhead irrigation. Rotate crops."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "chemical": "Apply Abamectin or Bifenazate miticide. Rotate chemicals to prevent resistance.",
        "organic":  "Spray strong water jets to dislodge mites. Use insecticidal soap or neem oil.",
        "prevention": "Maintain adequate plant watering. Avoid dusty conditions."
    },
    "Tomato___Target_Spot": {
        "chemical": "Apply Azoxystrobin or Pyraclostrobin fungicide at first sign of symptoms.",
        "organic":  "Remove infected leaves. Apply copper-based fungicide sprays.",
        "prevention": "Stake plants for better air circulation. Avoid overhead watering."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "chemical": "No cure. Control whitefly vector with Imidacloprid or Thiamethoxam.",
        "organic":  "Use yellow sticky traps for whiteflies. Apply neem oil or insecticidal soap.",
        "prevention": "Use virus-resistant varieties. Install insect-proof nets in nurseries."
    },
    "Tomato___Tomato_mosaic_virus": {
        "chemical": "No chemical cure. Remove and destroy infected plants immediately.",
        "organic":  "Wash hands and tools with soap before handling plants.",
        "prevention": "Use certified virus-free seeds. Control aphid vectors with neem oil."
    },
    "Tomato___healthy": {
        "chemical": "No treatment needed.",
        "organic":  "Continue balanced feeding and regular monitoring.",
        "prevention": "Rotate crops. Use mulch. Inspect plants weekly for early signs."
    },
}
SEV_ICON  = {"healthy":"✅","mild":"🟡","severe":"🔴"}
SEV_LABEL = {"healthy":"None","mild":"Moderate","severe":"Severe"}
SUPPORTED_CROPS = ["🍎 Apple","🫐 Blueberry","🍒 Cherry","🌽 Corn","🍇 Grape","🍊 Orange",
                   "🍑 Peach","🫑 Pepper","🥔 Potato","🍓 Raspberry","🌿 Soybean",
                   "🎃 Squash","🍓 Strawberry","🍅 Tomato"]

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_resources():
    tp = "models/crop_disease.tflite"
    np_ = "models/class_names.json"
    if not os.path.exists(tp):
        return None, None
    interp = tf.lite.Interpreter(model_path=tp)
    interp.allocate_tensors()
    with open(np_) as f:
        cn = json.load(f)
    labels = {i: n for i, n in enumerate(cn)} if isinstance(cn, list) else {int(k): v for k, v in cn.items()}
    return interp, labels

model, labels = load_resources()

def preprocess(img):
    img = img.resize((224,224)).convert("RGB")
    return np.expand_dims(np.array(img, dtype=np.float32)/255.0, 0)

def predict(interp, arr):
    i = interp.get_input_details()
    o = interp.get_output_details()
    interp.set_tensor(i[0]["index"], arr)
    interp.invoke()
    return interp.get_tensor(o[0]["index"])[0]

def find_info(key):
    if key in DISEASE_INFO: return DISEASE_INFO[key]
    kn = key.lower().replace(" ","").replace("-","").replace(",","").replace("_","")
    for k,v in DISEASE_INFO.items():
        if kn == k.lower().replace(" ","").replace("-","").replace(",","").replace("_",""):
            return v
    return None

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif!important;}
.stApp{background:#f8f9fb;}
#MainMenu,footer,header{visibility:hidden;}

/* ── Remove ALL Streamlit default padding/gaps ── */
.block-container{padding:0.5rem 1rem 1rem!important;}
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid #e2e6ee;}
[data-testid="stSidebar"]>div{padding:0!important;}
section[data-testid="stSidebar"] .block-container{padding:0.5rem 0.75rem!important;}
.stElementContainer{margin:0!important;}
div[data-testid="stVerticalBlock"]{gap:0.4rem!important;}

/* ── File uploader — fully working, custom styled ── */
[data-testid="stFileUploader"]{margin:0!important;padding:0!important;}
[data-testid="stFileUploader"] label{display:none!important;}
[data-testid="stFileUploaderDropzone"]{
  border:2px dashed #d1fae5!important;
  border-radius:12px!important;
  background:#f8fffe!important;
  padding:22px 16px 18px!important;
  text-align:center!important;
  cursor:pointer!important;
  transition:.2s!important;
  display:flex!important;
  flex-direction:column!important;
  align-items:center!important;
  justify-content:center!important;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:#10b981!important;
  background:#ecfdf5!important;
}
[data-testid="stFileUploaderDropzoneInstructions"]{display:none!important;}
[data-testid="stFileUploaderDropzone"] svg{display:none!important;}
[data-testid="stFileUploaderDropzone"] span{display:none!important;}
[data-testid="stFileUploaderDropzone"] small{display:none!important;}
[data-testid="stFileUploaderDropzone"] p{display:none!important;}
[data-testid="stFileUploaderDropzone"] button{
  background:linear-gradient(135deg,#059669,#10b981)!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  padding:8px 22px!important;font-size:12px!important;font-weight:700!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  margin-top:0!important;display:inline-block!important;
}
[data-testid="stFileUploaderFile"]{display:none!important;}
[data-testid="stFileUploaderFileData"]{display:none!important;}
[data-testid="stBaseButton-minimal"]{display:none!important;}
[data-testid="stFileUploader"] progress{display:none!important;}
div[data-testid="stFileUploader"] > div > div:nth-child(2){display:none!important;}
/* ── Icon and text injected BEFORE the dropzone button using CSS ── */
[data-testid="stFileUploaderDropzone"]::before{
  content:"🌿";
  font-size:30px;
  display:block;
  margin-bottom:8px;
}
[data-testid="stFileUploaderDropzone"]::after{
  content:"Upload Image\A Drop your leaf photo here\A JPG · PNG · JPEG";
  white-space:pre;
  display:block;
  text-align:center;
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:12px;
  color:#9aa3b5;
  line-height:1.8;
  margin-bottom:10px;
}
/* ── Upload box — custom styled ── */
.up-box{
  border:2px dashed #d1fae5;border-radius:12px;
  background:#f8fffe;text-align:center;
  padding:22px 16px 18px;cursor:pointer;transition:.2s;
}
.up-box:hover{border-color:#10b981;background:#ecfdf5;}
.up-box-icon{font-size:30px;margin-bottom:6px;}
.up-box-label{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#10b981;margin-bottom:4px;}
.up-box-title{font-size:13px;font-weight:700;color:#1e2535;margin-bottom:3px;}
.up-box-sub{font-size:11px;color:#9aa3b5;}

/* ── Sidebar label ── */
.lp-label{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:#9aa3b5;display:flex;align-items:center;gap:6px;margin-bottom:6px;}
.lp-label::before{content:'';width:12px;height:2px;background:#10b981;border-radius:1px;display:inline-block;}
.lp-div{height:1px;background:#e2e6ee;margin:14px 0;}



/* ── Tip items ── */
.tip-item{display:flex;gap:10px;padding:10px 12px;background:#f8f9fb;
  border:1px solid #e2e6ee;border-radius:9px;margin-bottom:8px;}
.tip-icon{font-size:15px;flex-shrink:0;margin-top:1px;}
.tip-text{font-size:12px;color:#5a6272;line-height:1.6;}
.tip-text strong{color:#1e2535;font-weight:700;}

/* ── Crop pills ── */
.crop-row{display:flex;flex-wrap:wrap;gap:6px;}
.crop-pill{padding:5px 12px;background:#fff;border:1px solid #e2e6ee;
  border-radius:100px;font-size:12px;font-weight:600;color:#5a6272;}

/* ── Topbar ── */
.topbar{background:#111827;border-radius:10px;padding:11px 20px;
  display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.tb-logo{display:flex;align-items:center;gap:9px;font-size:15px;font-weight:800;color:#fff;}
.tb-logo-ic{width:30px;height:30px;background:#10b981;border-radius:7px;
  display:flex;align-items:center;justify-content:center;font-size:15px;}
.tb-badge{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);
  color:#34d399;padding:3px 9px;border-radius:100px;}
.tb-right{font-size:11px;color:rgba(255,255,255,.5);display:flex;align-items:center;gap:6px;}
.live-dot{width:6px;height:6px;border-radius:50%;background:#10b981;
  display:inline-block;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}

/* ── Stats bar ── */
.stats-bar{display:flex;gap:8px;margin-bottom:10px;}
.stat-chip{background:#fff;border:1px solid #e2e6ee;border-radius:9px;
  padding:8px 12px;display:flex;align-items:center;gap:8px;flex:1;}
.stat-ic{font-size:16px;}
.stat-num{font-size:14px;font-weight:800;color:#111827;line-height:1;}
.stat-lbl{font-size:9px;color:#9aa3b5;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}

/* ── Analyze button ── */
.stButton>button{
  width:100%;padding:12px!important;
  background:linear-gradient(135deg,#059669,#10b981)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-size:13px!important;font-weight:800!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
}
.stButton>button:hover{box-shadow:0 6px 20px rgba(16,185,129,.3)!important;transform:translateY(-1px)!important;}
.stButton>button:disabled{background:#d1fae5!important;color:#6ee7b7!important;transform:none!important;box-shadow:none!important;}

/* ── Download buttons ── */
[data-testid="stDownloadButton"]>button{
  background:#fff!important;color:#1e2535!important;border:1.5px solid #e2e6ee!important;
  border-radius:9px!important;font-size:12px!important;font-weight:700!important;padding:10px!important;
}
[data-testid="stDownloadButton"]>button:hover{border-color:#10b981!important;color:#10b981!important;background:#ecfdf5!important;transform:none!important;box-shadow:none!important;}

/* ── Welcome state ── */
.welcome-box{background:#fff;border:1px solid #e2e6ee;border-radius:14px;
  padding:32px 24px;text-align:center;}
.welcome-icon{width:64px;height:64px;background:#ecfdf5;border-radius:16px;
  display:flex;align-items:center;justify-content:center;font-size:32px;margin:0 auto 14px;}
.welcome-title{font-size:18px;font-weight:800;color:#111827;margin-bottom:8px;}
.welcome-sub{font-size:13px;color:#9aa3b5;line-height:1.6;max-width:340px;margin:0 auto 14px;}

/* ── Result cards ── */
.res-head{background:#fff;border:1px solid #e2e6ee;border-radius:12px;
  padding:16px 18px;display:flex;align-items:center;gap:14px;margin-bottom:8px;}
.res-ic{width:48px;height:48px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-size:24px;flex-shrink:0;}
.res-ic.healthy{background:#ecfdf5;}.res-ic.mild{background:#fffbeb;}.res-ic.severe{background:#fef2f2;}
.res-crop{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#9aa3b5;margin-bottom:3px;}
.res-disease{font-size:18px;font-weight:800;color:#111827;margin-bottom:5px;}
.res-tag{font-size:9px;font-weight:700;padding:2px 8px;border-radius:100px;
  letter-spacing:.05em;text-transform:uppercase;display:inline-block;margin-right:4px;}
.tag-healthy{background:#ecfdf5;color:#059669;}.tag-mild{background:#fffbeb;color:#92400e;}
.tag-severe{background:#fef2f2;color:#991b1b;}.tag-crop{background:#eff4ff;color:#1e40af;}

/* ── Confidence card ── */
.conf-card{background:#fff;border:1px solid #e2e6ee;border-radius:12px;padding:16px 18px;margin-bottom:8px;}
.conf-head{font-size:11px;font-weight:700;color:#1e2535;margin-bottom:12px;display:flex;align-items:center;gap:7px;}
.conf-head span{background:#f0f2f6;padding:2px 7px;border-radius:5px;font-size:9px;color:#5a6272;}
.conf-row{margin-bottom:10px;}.conf-row:last-child{margin-bottom:0;}
.conf-meta{display:flex;justify-content:space-between;margin-bottom:4px;}
.conf-name{font-size:12px;font-weight:600;color:#1e2535;}
.conf-pct{font-size:12px;font-weight:700;}
.conf-track{height:7px;background:#f0f2f6;border-radius:100px;overflow:hidden;}
.conf-fill{height:100%;border-radius:100px;}
.fill-green{background:linear-gradient(90deg,#059669,#10b981);}
.fill-amber{background:linear-gradient(90deg,#d97706,#f59e0b);}
.fill-gray{background:#e2e6ee;}

/* ── Info grid ── */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;}
.info-card{background:#fff;border:1px solid #e2e6ee;border-radius:11px;padding:13px 15px;}
.ic-lbl{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#9aa3b5;margin-bottom:4px;}
.ic-val{font-size:13px;font-weight:700;color:#111827;line-height:1.3;}
.ic-sub{font-size:10px;color:#5a6272;margin-top:2px;}

/* ── Treatment card ── */
.treat-card{background:#fff;border:1px solid #e2e6ee;border-radius:12px;padding:16px 18px;margin-bottom:8px;}
.treat-title{font-size:11px;font-weight:700;color:#1e2535;margin-bottom:12px;display:flex;align-items:center;gap:7px;}
.treat-title span{background:#f0f2f6;padding:2px 7px;border-radius:5px;font-size:9px;color:#5a6272;}
.treat-row{display:flex;gap:10px;margin-bottom:10px;}
.treat-row:last-child{margin-bottom:0;}
.treat-ic{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.treat-ic.chem{background:#fef2f2;}
.treat-ic.org{background:#ecfdf5;}
.treat-ic.prev{background:#eff4ff;}
.treat-content{flex:1;}
.treat-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px;}
.treat-label.chem{color:#991b1b;}
.treat-label.org{color:#059669;}
.treat-label.prev{color:#1e40af;}
.treat-text{font-size:12px;color:#5a6272;line-height:1.55;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="lp-label" style="margin-top:14px;">Upload Leaf Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed", key=f"uploader_{st.session_state.uploader_key}")

    if uploaded:
        st.markdown('<style>[data-testid="stFileUploaderDropzone"]{display:none!important;}</style>', unsafe_allow_html=True)
        pil_img = Image.open(uploaded)
        st.markdown('<div style="border:2px solid #d1fae5;border-radius:12px;overflow:hidden;">', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown(
            '<p style="font-size:11px;color:#5a6272;text-align:center;padding:4px 0 6px;background:#fff;margin:0;">📄 '
            + uploaded.name + '</p></div>',
            unsafe_allow_html=True
        )

    # Track new file → reset result
    if uploaded:
        fid = uploaded.name + str(uploaded.size)
        if st.session_state.get("last_fid") != fid:
            st.session_state.show_result = False
            st.session_state.pop("last_conf", None)
            st.session_state.last_fid = fid
    else:
        st.session_state.show_result = False
        st.session_state.pop("last_fid", None)

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    analyze = st.button("🔍  Analyze Disease", disabled=(uploaded is None), use_container_width=True)
    if analyze and uploaded:
        st.session_state.show_result = True

    st.markdown('<div class="lp-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-label">Tips</div>', unsafe_allow_html=True)
    for ic, txt in [
        ("☀️","<strong>Good lighting</strong> — Natural daylight works best"),
        ("🔍","<strong>Close-up shot</strong> — Fill frame with the leaf"),
        ("🌿","<strong>Single leaf</strong> — One leaf per photo"),
        ("📷","<strong>In focus</strong> — Blurry images reduce accuracy"),
    ]:
        st.markdown('<div class="tip-item"><div class="tip-icon">'+ic+'</div><div class="tip-text">'+txt+'</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="lp-div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-label">Supported Crops</div>', unsafe_allow_html=True)
    st.markdown('<div class="crop-row">'+"".join('<span class="crop-pill">'+c+'</span>' for c in SUPPORTED_CROPS)+'</div>', unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────
# Topbar
st.markdown("""
<div class="topbar">
  <div class="tb-logo"><div class="tb-logo-ic">🌿</div>CropGuard AI</div>
  <div class="tb-badge">Deep Learning · MobileNetV2</div>
  <div class="tb-right"><span class="live-dot"></span> Model ready</div>
</div>""", unsafe_allow_html=True)

# Stats bar
conf_disp = st.session_state.get("last_conf","—")
st.markdown(
    '<div class="stats-bar">'
    '<div class="stat-chip"><span class="stat-ic">🧠</span><div><div class="stat-num">38</div><div class="stat-lbl">Disease Classes</div></div></div>'
    '<div class="stat-chip"><span class="stat-ic">🌱</span><div><div class="stat-num">14</div><div class="stat-lbl">Crop Types</div></div></div>'
    '<div class="stat-chip"><span class="stat-ic">⚡</span><div><div class="stat-num">MobileNetV2</div><div class="stat-lbl">Architecture</div></div></div>'
    '<div class="stat-chip"><span class="stat-ic">🎯</span><div><div class="stat-num">'+conf_disp+'</div><div class="stat-lbl">Confidence</div></div></div>'
    '</div>', unsafe_allow_html=True)

# ── State: No image ──
if not uploaded:
    dpills = "".join('<span class="crop-pill">'+d+'</span>' for d in
        ["Early Blight","Late Blight","Leaf Mold","Mosaic Virus","Powdery Mildew","Rust","Bacterial Spot","+ more"])
    st.markdown(
        '<div class="welcome-box">'
        '<div class="welcome-icon">🌾</div>'
        '<div class="welcome-title">Upload a leaf to get started</div>'
        '<div class="welcome-sub">Our deep learning model analyses your crop leaf and identifies diseases instantly.</div>'
        '<div class="crop-row" style="justify-content:center;">'+dpills+'</div>'
        '</div>', unsafe_allow_html=True)

# ── State: Image loaded, not yet analysed ──
elif not st.session_state.show_result:
    st.markdown(
        '<div class="welcome-box" style="padding:28px 24px;">'
        '<div style="font-size:32px;margin-bottom:10px">👆</div>'
        '<div class="welcome-title" style="font-size:16px;">Click <span style="color:#10b981">Analyze Disease</span></div>'
        '<div class="welcome-sub">Image loaded — press the button in the sidebar to run the model.</div>'
        '</div>', unsafe_allow_html=True)

# ── State: Show result ──
else:
    if model is None:
        st.error("⚠️ Model not found. Make sure models/crop_disease.tflite and models/class_names.json exist.")
        st.stop()

    with st.spinner("🔍 Analysing..."):
        time.sleep(0.2)
        preds = predict(model, preprocess(Image.open(uploaded)))

    top3  = np.argsort(preds)[::-1][:3]
    conf1 = float(preds[top3[0]])*100
    raw   = labels[top3[0]]
    info  = find_info(raw) or {
        "crop": raw.split("___")[0].replace("_"," ").title(),
        "disease": raw.split("___")[-1].replace("_"," ").title() if "___" in raw else raw,
        "sev":"mild","area":"20–40%","urgency":"Consult expert","season":"Warm months"
    }
    sev, crop, disease = info["sev"], info["crop"], info["disease"]
    icon, sev_lbl = SEV_ICON[sev], SEV_LABEL[sev]
    st.session_state["last_conf"] = f"{conf1:.1f}%"

    # Result header
    st.markdown(
        '<div class="res-head">'
        '<div class="res-ic '+sev+'">'+icon+'</div>'
        '<div>'
        '<div class="res-crop">'+crop+'</div>'
        '<div class="res-disease">'+disease+'</div>'
        '<span class="res-tag tag-'+sev+'">'+icon+' '+sev_lbl+'</span>'
        '<span class="res-tag tag-crop">🌱 '+crop+'</span>'
        '</div></div>', unsafe_allow_html=True)

    # Confidence bars
    fills  = ["fill-green","fill-amber","fill-gray"]
    colors = ["#059669","#d97706","#9aa3b5"]
    html   = '<div class="conf-card"><div class="conf-head">Prediction Confidence <span>Top 3</span></div>'
    for r, idx in enumerate(top3):
        inf2 = find_info(labels[idx]) or {}
        dn   = inf2.get("disease", labels[idx].split("___")[-1].replace("_"," ").title() if "___" in labels[idx] else labels[idx])
        cp   = inf2.get("crop",    labels[idx].split("___")[0].replace("_"," ").title())
        pct  = str(round(float(preds[idx])*100, 1))+"%"
        html += (
            '<div class="conf-row"><div class="conf-meta">'
            '<span class="conf-name">'+cp+' — '+dn+'</span>'
            '<span class="conf-pct" style="color:'+colors[r]+'">'+pct+'</span>'
            '</div><div class="conf-track"><div class="conf-fill '+fills[r]+'" style="width:'+pct+'"></div></div></div>'
        )
    st.markdown(html+'</div>', unsafe_allow_html=True)

    # Info grid
    st.markdown(
        '<div class="info-grid">'
        '<div class="info-card"><div class="ic-lbl">Severity Level</div><div class="ic-val">'+sev_lbl+'</div><div class="ic-sub">'+("No disease" if sev=="healthy" else "Needs attention")+'</div></div>'
        '<div class="info-card"><div class="ic-lbl">Affected Area</div><div class="ic-val">'+info["area"]+'</div><div class="ic-sub">Estimated leaf surface</div></div>'
        '<div class="info-card"><div class="ic-lbl">Urgency</div><div class="ic-val">'+info["urgency"]+'</div><div class="ic-sub">Recommended action</div></div>'
        '<div class="info-card"><div class="ic-lbl">Common In</div><div class="ic-val">'+info["season"]+'</div><div class="ic-sub">Peak occurrence</div></div>'
        '</div>', unsafe_allow_html=True)

    # Treatment card
    treat = TREATMENT_INFO.get(raw)
    if not treat:
        for k in TREATMENT_INFO:
            kn = k.lower().replace(" ","").replace("_","")
            rn = raw.lower().replace(" ","").replace("_","")
            if kn == rn:
                treat = TREATMENT_INFO[k]
                break
    if treat and sev != "healthy":
        st.markdown(
            '<div class="treat-card">'
            '<div class="treat-title">How to Treat <span>3 approaches</span></div>'
            '<div class="treat-row">'
            '<div class="treat-ic chem">💊</div>'
            '<div class="treat-content"><div class="treat-label chem">Chemical</div>'
            '<div class="treat-text">' + treat["chemical"] + '</div></div></div>'
            '<div class="treat-row">'
            '<div class="treat-ic org">🌿</div>'
            '<div class="treat-content"><div class="treat-label org">Organic</div>'
            '<div class="treat-text">' + treat["organic"] + '</div></div></div>'
            '<div class="treat-row">'
            '<div class="treat-ic prev">🛡️</div>'
            '<div class="treat-content"><div class="treat-label prev">Prevention</div>'
            '<div class="treat-text">' + treat["prevention"] + '</div></div></div>'
            '</div>',
            unsafe_allow_html=True
        )

    # Action buttons
    treat_lines = ""
    if treat and sev != "healthy":
        treat_lines = f"\nChemical: {treat['chemical']}\nOrganic: {treat['organic']}\nPrevention: {treat['prevention']}"
    report = f"CropGuard AI Report\nCrop: {crop}\nDisease: {disease}\nSeverity: {sev_lbl}\nConfidence: {conf1:.1f}%\nArea: {info['area']}\nUrgency: {info['urgency']}\nSeason: {info['season']}{treat_lines}\n"
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄  New Analysis", use_container_width=True):
            st.session_state.show_result = False
            st.session_state.uploader_key += 1
            st.session_state.pop("last_conf", None)
            st.session_state.pop("last_fid", None)
            st.rerun()
    with c2:
        st.download_button("📋  Download Report", report, file_name="cropguard_report.txt", mime="text/plain", use_container_width=True)
    with c3:
        st.download_button("🖼️  Save Image", uploaded.getvalue(), file_name=disease.replace(" ","_")+".jpg", mime="image/jpeg", use_container_width=True)
