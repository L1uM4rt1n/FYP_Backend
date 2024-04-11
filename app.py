import pandas as pd
import numpy as np
import datetime
from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import joblib
import os


class CustomEnsemble:
    def __init__(self, model_paths, weights):
        self.models = [joblib.load(path) for path in model_paths]
        self.weights = weights

    def predict_proba(self, X):
        predictions = np.array([model.predict_proba(X) for model in self.models])
        weighted_sum = np.tensordot(predictions, self.weights, axes=((0), (0)))
        return weighted_sum

    def predict(self, X):
        weighted_sum = self.predict_proba(X)
        return np.argmax(weighted_sum, axis=1)

    def evaluate(self, X, y):
        weighted_sum = self.predict_proba(X)
        final_predictions = self.predict(X)


VISIT_ID = "visits.visit_id"
PATIENT_ID = "patients.patient_id"

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://b5cf0d40805bd4:927befde@us-cluster-east-01.k8s.cleardb.net/heroku_ea7263fb720321f?reconnect=true" # heroku deployment
# app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:LZWMadh7187%40%400100@localhost/triagedb"  # martin

db = SQLAlchemy(app)

app_root = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(app_root, "linear_combi.pkl")
with open(model_path, "rb") as model_file:
    model = joblib.load(model_file)

scaler_path = os.path.join(app_root, "scaler.pkl")
with open(scaler_path, "rb") as scaler_file:
    scaler = joblib.load(scaler_file)


class Patient(db.Model):
    __tablename__ = "patients"
    patient_id = db.Column(db.String(255), primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(255))
    phone_number = db.Column(db.String(255), unique=True)


class Visit(db.Model):
    __tablename__ = "visits"
    visit_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(255), db.ForeignKey("patients.patient_id"), primary_key=True)
    date_of_visit = db.Column(db.Date)
    time_of_visit = db.Column(db.Time)


class Vital(db.Model):
    __tablename__ = "vitals"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    heart_rate = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    systolic_bp = db.Column(db.Integer)
    resp_rate = db.Column(db.Integer)
    pain_score = db.Column(db.Integer)
    spo2 = db.Column(db.Integer)
    temp = db.Column(db.Float)
    avpu_score = db.Column(db.String(255))
    adjusted_si = db.Column(db.Float)
    mews_score = db.Column(db.Integer)
    mode_of_arrival = db.Column(db.String(255))


class MedicalCondition(db.Model):
    __tablename__ = "medical_conditions"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    has_allergy = db.Column(db.Integer)
    has_diabetes = db.Column(db.Integer)
    has_hyperlipidemia = db.Column(db.Integer)
    has_hypertension = db.Column(db.Integer)


class HistoricalAbdominalIllness(db.Model):
    __tablename__ = "historical_abdominal_illness"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    has_abdominal_cancer = db.Column(db.Integer)
    has_chronic_kidney_disease = db.Column(db.Integer)
    has_gallstone = db.Column(db.Integer)
    has_gastric_problem = db.Column(db.Integer)
    has_gastroesophageal_reflux_disease = db.Column(db.Integer)
    has_irritable_bowel_syndrome = db.Column(db.Integer)
    has_kidney_stones = db.Column(db.Integer)
    has_liver_disease = db.Column(db.Integer)


class Symptom(db.Model):
    __tablename__ = "symptoms"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    has_fever = db.Column(db.Integer)
    has_nausea = db.Column(db.Integer)
    has_vomiting = db.Column(db.Integer)
    has_diarrhoea = db.Column(db.Integer)
    has_dysuria = db.Column(db.Integer)
    has_abdominal_bloating = db.Column(db.Integer)
    has_back_pain = db.Column(db.Integer)
    has_chest_pain = db.Column(db.Integer)
    has_constipation = db.Column(db.Integer)
    has_difficulty_urinating = db.Column(db.Integer)
    has_fatigue = db.Column(db.Integer)
    has_gastric_pain = db.Column(db.Integer)
    has_gastritis = db.Column(db.Integer)
    has_heartburn = db.Column(db.Integer)
    has_appetite_loss = db.Column(db.Integer)
    has_shivers = db.Column(db.Integer)
    has_weight_loss = db.Column(db.Integer)
    has_jaundice = db.Column(db.Integer)


class FemalePatientInfo(db.Model):
    __tablename__ = "female_patient_info"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    last_menstrual_cycle = db.Column(db.Integer)
    pregnant_yes = db.Column(db.Integer)
    pregnant_no = db.Column(db.Integer)
    pregnant_unsure = db.Column(db.Integer)
    has_vagina_bleeding = db.Column(db.Integer)


class RemainingQuestion(db.Model):
    __tablename__ = "remaining_questions"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    has_past_abdominal_surgery = db.Column(db.Integer)
    blood_thinning_medication = db.Column(db.Integer)
    is_alcohol_drinker = db.Column(db.Integer)
    is_smoker = db.Column(db.Integer)
    pain_location = db.Column(db.String(255))
    nature_of_pain = db.Column(db.String(255))
    colour_of_stool = db.Column(db.String(255))
    contents_of_vomit = db.Column(db.String(255))
    has_blood_in_stool = db.Column(db.Integer)
    has_loose_stool = db.Column(db.Integer)
    had_bbq_steamboat_rawFood_spicyFood_overnightLeftovers = db.Column(db.Integer)
    pain_after_meal = db.Column(db.Integer)


class TriageResult(db.Model):
    __tablename__ = "triage_results"
    patient_id = db.Column(db.String(255), db.ForeignKey(PATIENT_ID), primary_key=True)
    visit_id = db.Column(db.Integer, db.ForeignKey(VISIT_ID), primary_key=True)
    triage_timestamp = db.Column(db.DateTime, default=func.now(), primary_key=True)
    triage_class = db.Column(db.Integer)


@app.route("/submit_patient", methods=["POST"])
def submit_patient():
    data = request.get_json()

    existing_patient = Patient.query.filter_by(patient_id=data.get("patient_id")).first()

    if not existing_patient:
        existing_patient = Patient(
            patient_id=data.get("patient_id"),
            age=data.get("age"),
            gender=data.get("gender"),
            phone_number=data.get("phone_number")
        )
        db.session.add(existing_patient)
        db.session.flush()

    existing_visits = Visit.query.filter_by(patient_id=existing_patient.patient_id).count()

    visit = Visit(
        patient_id=existing_patient.patient_id,
        date_of_visit=datetime.datetime.now().date(),
        time_of_visit=datetime.datetime.now().time()
    )

    if existing_visits == 0:
        visit.visit_id = 1
    else:
        visit.visit_id = existing_visits + 1

    try:
        db.session.add(visit)
        db.session.commit()
        return {"message": "Patient and Visit tables updated successfully"}
    except Exception as e:
        db.session.rollback()
        return {"error": str(e)}, 500


@app.route("/get_latest_visit/<string:patient_id>", methods=["GET"])
def get_latest_visit(patient_id):
    latest_visit = Visit.query.filter_by(patient_id=patient_id).order_by(Visit.visit_id.desc()).first()
    if latest_visit is None:
        return {"error": "No visits found for this patient"}, 404

    return {"visit_id": latest_visit.visit_id}


@app.route("/submit_vitals", methods=["POST"])
def submit_vitals():
    data = request.get_json()
    get_patient_id = data.get("patient_id")
    get_visit_id = data.get("visit_id")

    vitals = db.session.query(Vital).filter(Vital.patient_id == get_patient_id, Vital.visit_id == get_visit_id).first()

    if vitals is None:
        vitals = Vital(
            patient_id=get_patient_id,
            visit_id=get_visit_id,
            heart_rate=data.get("heartRate"),
            diastolic_bp=data.get("diastolicBP"),
            systolic_bp=data.get("systolicBP"),
            resp_rate=data.get("respRate"),
            pain_score=data.get("painScore"),
            spo2=data.get("spo2"),
            temp=data.get("temp"),
            avpu_score=data.get("avpuScore"),
            adjusted_si=data.get("adjustedSI"),
            mews_score=data.get("mewsScore"),
            mode_of_arrival=data.get("modeArrival")
        )
        db.session.add(vitals)
    else:
        vitals.heart_rate = data.get("heartRate")
        vitals.diastolic_bp = data.get("diastolicBP")
        vitals.systolic_bp = data.get("systolicBP")
        vitals.resp_rate = data.get("respRate")
        vitals.pain_score = data.get("painScore")
        vitals.spo2 = data.get("spo2")
        vitals.temp = data.get("temp")
        vitals.avpu_score = data.get("avpuScore")
        vitals.adjusted_si = data.get("adjustedSI")
        vitals.mews_score = data.get("mewsScore")
        vitals.mode_of_arrival = data.get("modeArrival")
    try:
        db.session.commit()
        return {"message": "Vitals submitted successfully"}
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


@app.route("/update_vitals", methods=["POST"])
def update_vitals():
    data = request.get_json()
    get_patient_id = data.get("patient_id")
    get_visit_id = data.get("visit_id")

    vitals = db.session.query(Vital).filter(Vital.patient_id == get_patient_id, Vital.visit_id == get_visit_id).first()

    if vitals is None:
        return {
            "error": f"Vitals not found for the given patient ID: {get_patient_id} and visit ID: {get_visit_id}"}, 404

    vitals.heart_rate = data.get("heartRate")
    vitals.diastolic_bp = data.get("diastolicBP")
    vitals.systolic_bp = data.get("systolicBP")
    vitals.resp_rate = data.get("respRate")
    vitals.pain_score = data.get("painScore")
    vitals.spo2 = data.get("spo2")
    vitals.temp = data.get("temp")
    vitals.avpu_score = data.get("avpuScore")
    vitals.adjusted_si = data.get("adjustedSI")
    vitals.mews_score = data.get("mewsScore")
    vitals.mode_of_arrival = data.get("modeArrival")

    try:
        db.session.commit()
        return {"message": "Vitals updated successfully"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 500


@app.route("/get_vitals_info/<string:patient_id>/<int:visit_id>", methods=["GET"])
def get_vitals_info(patient_id, visit_id):
    vitals = db.session.query(Vital).filter(Vital.patient_id == patient_id, Vital.visit_id == visit_id).first()
    if vitals is None:
        return {"error": "Vitals not found for the given patient ID and visit ID"}, 404

    vitals_info = {
        "heartRate": vitals.heart_rate,
        "diastolicBP": vitals.diastolic_bp,
        "systolicBP": vitals.systolic_bp,
        "respRate": vitals.resp_rate,
        "painScore": vitals.pain_score,
        "spo2": vitals.spo2,
        "temp": vitals.temp,
        "avpuScore": vitals.avpu_score,
        "adjustedSI": vitals.adjusted_si,
        "mewsScore": vitals.mews_score,
        "modeArrival": vitals.mode_of_arrival
    }

    return vitals_info


def convert_to_model_format(patient_data):
    data = {}

    mode_of_arrival_mapping = {
        "own": 0,
        "privAmbulance": 1,
        "scdf": 2
    }

    patient_gender_mapping = {
        "m": 0,
        "f": 1
    }

    for patient, vital, medical_condition, historical_abdominal_illness, symptom, female_patient_info, remaining_question in patient_data:
        data["Mode of Arrival"] = mode_of_arrival_mapping.get(vital.mode_of_arrival)
        data["Patient Age"] = patient.age
        data["Patient Gender"] = patient_gender_mapping.get(patient.gender)
        data["Blood Pressure, Non-Invasive, Systolic (mmHg)"] = vital.systolic_bp
        data["Respiration Rate"] = vital.resp_rate
        data["Pain Score at presentation"] = vital.pain_score
        data["SpO2 (%)"] = vital.spo2
        data["Blood Pressure, Non-Invasive, Diastolic (mmHg)"] = vital.diastolic_bp
        data["Temperature (deg. C)"] = vital.temp
        data["Heart Rate (beats/min)"] = vital.heart_rate
        data["menstruation"] = female_patient_info.last_menstrual_cycle
        data["pregnant_yes"] = female_patient_info.pregnant_yes
        data["pregnant_no"] = female_patient_info.pregnant_no
        data["pregnant_unsure"] = female_patient_info.pregnant_unsure
        data["bleeding from vagina"] = female_patient_info.has_vagina_bleeding
        data["abdominal surgery"] = remaining_question.has_past_abdominal_surgery
        data["hypertension"] = medical_condition.has_hypertension
        data["diabetes mellitus"] = medical_condition.has_diabetes
        data["hyperlipidemia"] = medical_condition.has_hyperlipidemia
        data["blood thinner"] = remaining_question.blood_thinning_medication
        data["abdominal cancer"] = historical_abdominal_illness.has_abdominal_cancer
        data["chronic kidney disease"] = historical_abdominal_illness.has_chronic_kidney_disease
        data["gallstone"] = historical_abdominal_illness.has_gallstone
        data["gastric history"] = historical_abdominal_illness.has_gastric_problem
        data["gastroesophageal reflux disease"] = historical_abdominal_illness.has_gastroesophageal_reflux_disease
        data["irritable bowel syndrome"] = historical_abdominal_illness.has_irritable_bowel_syndrome
        data["kidney stone"] = historical_abdominal_illness.has_kidney_stones
        data["liver disease"] = historical_abdominal_illness.has_liver_disease
        data["smoker"] = remaining_question.is_smoker
        data["alcohol drinker"] = remaining_question.is_alcohol_drinker
        ##################### Pain Locations #####################
        pain_locations = remaining_question.pain_location.split(", ")
        data["right hypochondrium pain"] = 1 if "right hypochondrium pain" in pain_locations else 0
        data["epigastric pain"] = 1 if "epigastric pain" in pain_locations else 0
        data["left hypochondrium pain"] = 1 if "left hypochondrium pain" in pain_locations else 0
        data["right lumbar pain"] = 1 if "right lumbar pain" in pain_locations else 0
        data["umbilical pain"] = 1 if "umbilical pain" in pain_locations else 0
        data["left lumbar pain"] = 1 if "left lumbar pain" in pain_locations else 0
        data["right iliac fossa pain"] = 1 if "right iliac fossa pain" in pain_locations else 0
        data["hypogastric pain"] = 1 if "hypogastric pain" in pain_locations else 0
        data["left iliac fossa pain"] = 1 if "left iliac fossa pain" in pain_locations else 0
        ##################### Pain Locations #####################
        ##################### Nature of Pain #####################
        nature_of_pains = remaining_question.nature_of_pain.split(", ")
        data["crampy pain"] = 1 if "Crampy" in nature_of_pains else 0
        data["radiating pain"] = 1 if "Radiating" in nature_of_pains else 0
        data["sharp pain"] = 1 if "Sharp" in nature_of_pains else 0
        data["squeezing pain"] = 1 if "Squeezing" in nature_of_pains else 0
        data["stabbing pain"] = 1 if "Stabbing" in nature_of_pains else 0
        data["colicky pain"] = 1 if "Pain that Comes & Goes" in nature_of_pains else 0
        data["persistent pain"] = 1 if "Persistent" in nature_of_pains else 0
        data["pulling pain"] = 1 if "Pulling" in nature_of_pains else 0
        data["moderate pain"] = 1 if "Moderate" in nature_of_pains else 0
        data["severe pain"] = 1 if "Severe" in nature_of_pains else 0
        ##################### Nature of Pain #####################
        data["fever"] = symptom.has_fever
        data["nausea"] = symptom.has_nausea
        data["vomit"] = symptom.has_vomiting
        data["diarrhea"] = symptom.has_diarrhoea
        data["dysuria"] = symptom.has_dysuria
        data["abdominal bloatedness"] = symptom.has_abdominal_bloating
        data["back pain"] = symptom.has_back_pain
        data["constipation"] = symptom.has_constipation
        data["chest pain"] = symptom.has_chest_pain
        data["urinary hesitancy"] = symptom.has_difficulty_urinating
        data["fatigue"] = symptom.has_fatigue
        data["gastric pain"] = symptom.has_gastric_pain
        data["gastritis"] = symptom.has_gastritis
        data["heartburn"] = symptom.has_heartburn
        data["appetite poor"] = symptom.has_appetite_loss
        data["shivers"] = symptom.has_shivers
        data["loss of weight"] = symptom.has_weight_loss
        data["jaundice"] = symptom.has_jaundice
        ##################### Contents of Vomit #####################
        contents_of_vomit = remaining_question.contents_of_vomit.split(", ")
        data["blood vomit"] = 1 if "Blood" in contents_of_vomit else 0
        data["coffee-ground vomit"] = 1 if "Coffee-ground like" in contents_of_vomit else 0
        data["yellow vomit"] = 1 if "Yellow" in contents_of_vomit else 0
        data["green vomit"] = 1 if "Green" in contents_of_vomit else 0
        data["vomited undigested food"] = 1 if "Undigested Food" in contents_of_vomit else 0
        ##################### Contents of Vomit #####################
        ##################### Colour of Stool #####################
        colour_of_stool = remaining_question.colour_of_stool.split(", ")
        data["pale stool"] = 1 if "Pale" in colour_of_stool else 0
        data["yellow stool"] = 1 if "Yellow" in colour_of_stool else 0
        data["brown stool"] = 1 if "Brown" in colour_of_stool else 0
        data["green stool"] = 1 if "Green" in colour_of_stool else 0
        data["black stool"] = 1 if "Black" in colour_of_stool else 0
        ##################### Colour of Stool #####################
        data["loose stool"] = remaining_question.has_loose_stool
        data["bloody stool"] = remaining_question.has_blood_in_stool
        data["bad food"] = remaining_question.had_bbq_steamboat_rawFood_spicyFood_overnightLeftovers
        data["pain after bad food"] = remaining_question.pain_after_meal

    print(data)

    return data


def predict_triage(patient_id, visit_id):
    patient_data = db.session.query(
        Patient,
        Vital,
        MedicalCondition,
        HistoricalAbdominalIllness,
        Symptom,
        FemalePatientInfo,
        RemainingQuestion
    ).filter(Vital.patient_id == patient_id, Vital.visit_id == visit_id).all()

    data = convert_to_model_format(patient_data)
    data_df = pd.DataFrame(data, index=[0])

    continuous_features = ['Patient Age', 'Blood Pressure, Non-Invasive, Diastolic (mmHg)',
                           'Blood Pressure, Non-Invasive, Systolic (mmHg)',
                           'Heart Rate (beats/min)', 'Pain Score at presentation',
                           'Respiration Rate', 'SpO2 (%)', 'Temperature (deg. C)']

    data_df[continuous_features] = scaler.transform(data_df[continuous_features])

    result = model.predict(data_df)
    return result[0].tolist()


@app.route("/submit_features", methods=["POST"])
def submit_features():
    data = request.get_json()

    get_patient_id = data.get("patient_id")
    get_visit_id = data.get("visit_id")

    medical_condition = MedicalCondition(
        patient_id=get_patient_id,
        visit_id=get_visit_id,
        has_allergy=data.get("has_allergy"),
        has_diabetes=data.get("diabetes mellitus"),
        has_hyperlipidemia=data.get("hyperlipidemia"),
        has_hypertension=data.get("hypertension")
    )

    historical_abdominal_illness = HistoricalAbdominalIllness(
        patient_id=get_patient_id,
        visit_id=get_visit_id,
        has_abdominal_cancer=data.get("has_abdominal_cancer"),
        has_chronic_kidney_disease=data.get("chronic kidney disease"),
        has_gallstone=data.get("gallstone"),
        has_gastric_problem=data.get("gastric problem"),
        has_gastroesophageal_reflux_disease=data.get("gastroesophageal reflux disease"),
        has_irritable_bowel_syndrome=data.get("irritable bowel syndrome"),
        has_kidney_stones=data.get("has_kidney_stones"),
        has_liver_disease=data.get("liver disease")
    )

    symptom = Symptom(
        patient_id=get_patient_id,
        visit_id=get_visit_id,
        has_fever=data.get("fever"),
        has_nausea=data.get("nausea"),
        has_vomiting=data.get("vomit"),
        has_diarrhoea=data.get("diarrhea"),
        has_dysuria=data.get("dysuria"),
        has_abdominal_bloating=data.get("abdominal bloatedness"),
        has_back_pain=data.get("back pain"),
        has_chest_pain=data.get("chest pain"),
        has_constipation=data.get("constipation"),
        has_difficulty_urinating=data.get("difficulty urinating"),
        has_fatigue=data.get("has_fatigue"),
        has_gastric_pain=data.get("gastric pain"),
        has_gastritis=data.get("gastritis"),
        has_heartburn=data.get("has_heartburn"),
        has_appetite_loss=data.get("loss of appetite"),
        has_shivers=data.get("shivers"),
        has_weight_loss=data.get("loss of weight"),
        has_jaundice=data.get("jaundice")
    )

    female_patient_info = FemalePatientInfo(
        patient_id=get_patient_id,
        visit_id=get_visit_id,
        last_menstrual_cycle=data.get("last_menstrual_cycle"),
        pregnant_yes=data.get("pregnant_yes"),
        pregnant_no=data.get("pregnant_no"),
        pregnant_unsure=data.get("pregnant_unsure"),
        has_vagina_bleeding=data.get("has_vagina_bleeding")
    )

    remaining_question = RemainingQuestion(
        patient_id=get_patient_id,
        visit_id=get_visit_id,
        has_past_abdominal_surgery=data.get("has_past_abdominal_surgery"),
        blood_thinning_medication=data.get("consumes blood thinning med"),
        is_alcohol_drinker=data.get("is_alcohol_drinker"),
        is_smoker=data.get("is_smoker"),
        pain_location=data.get("pain_location"),
        nature_of_pain=data.get("nature_of_pain"),
        colour_of_stool=data.get("stool_colour"),
        contents_of_vomit=data.get("contents_of_vomit"),
        has_blood_in_stool=data.get("has_bloody_stool"),
        has_loose_stool=data.get("has_loose_stool"),
        had_bbq_steamboat_rawFood_spicyFood_overnightLeftovers=data.get("has_barbeque_steamboat_raw_food_spicy_food"),
        pain_after_meal=data.get("pain_after_meal")
    )

    try:
        db.session.bulk_save_objects(
            [medical_condition, historical_abdominal_illness, symptom, female_patient_info, remaining_question])
        db.session.commit()

        prediction = predict_triage(get_patient_id, get_visit_id)

        results_checker = db.session.query(TriageResult).filter(TriageResult.patient_id == get_patient_id,
                                                                TriageResult.visit_id == get_visit_id).first()
        if results_checker is not None:
            db.session.delete(results_checker)
            db.session.commit()

        triage_result = TriageResult(
            patient_id=get_patient_id,
            visit_id=get_visit_id,
            triage_timestamp=func.now(),
            triage_class=prediction + 1
        )

        db.session.add(triage_result)
        db.session.commit()
        return {"message": "Features and triage result submitted successfully", "triageResult": prediction}
    except Exception as e:
        print(str(e))
        return {"error": str(e)}, 500


@app.route("/update_vitals_for_prediction", methods=["POST"])
def update_vitals_for_prediction():
    data = request.get_json()
    get_patient_id = data.get("patient_id")
    get_visit_id = data.get("visit_id")

    vitals = db.session.query(Vital).filter(Vital.patient_id == get_patient_id, Vital.visit_id == get_visit_id).first()

    if vitals is None:
        return {
            "error": f"Vitals not found for the given patient ID: {get_patient_id} and visit ID: {get_visit_id}"}, 404

    vitals.heart_rate = data.get("heartRate")
    vitals.diastolic_bp = data.get("diastolicBP")
    vitals.systolic_bp = data.get("systolicBP")
    vitals.resp_rate = data.get("respRate")
    vitals.pain_score = data.get("painScore")
    vitals.spo2 = data.get("spo2")
    vitals.temp = data.get("temp")
    vitals.avpu_score = data.get("avpuScore")
    vitals.adjusted_si = data.get("adjustedSI")
    vitals.mews_score = data.get("mewsScore")
    vitals.mode_of_arrival = data.get("modeArrival")

    try:
        db.session.commit()
        prediction = predict_triage(get_patient_id, get_visit_id)

        results_checker = db.session.query(TriageResult).filter(TriageResult.patient_id == get_patient_id,
                                                                TriageResult.visit_id == get_visit_id).first()
        if results_checker is not None:
            triage_result = TriageResult(
                patient_id=get_patient_id,
                visit_id=get_visit_id,
                triage_timestamp=func.now(),
                triage_class=prediction + 1
            )

            db.session.add(triage_result)
        else:
            results_checker.triage_timestamp = func.now()
            results_checker.triage_class = prediction + 1

        db.session.commit()
        return {"message": "Vitals updated successfully with the triage result: " + str(prediction + 1)}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 500


@app.route("/update_pregnancy_status", methods=["POST"])
def update_pregnancy_status():
    data = request.get_json()

    patient_id = data.get("patient_id")
    visit_id = data.get("visit_id")
    pregnancy_yes = data.get("pregnancy_yes")
    pregnancy_no = data.get("pregnancy_no")
    pregnancy_unsure = data.get("pregnant_unsure")

    female_patient_info = db.session.query(FemalePatientInfo).filter(FemalePatientInfo.patient_id == patient_id,
                                                                     FemalePatientInfo.visit_id == visit_id).first()

    if female_patient_info is None:
        return {"error": "Female patient info not found for the given patient ID and visit ID"}, 404
    else:
        female_patient_info.pregnant_yes = pregnancy_yes
        female_patient_info.pregnant_no = pregnancy_no
        female_patient_info.pregnant_unsure = pregnancy_unsure

    try:
        db.session.commit()
        prediction = predict_triage(patient_id, visit_id)

        results_checker = db.session.query(TriageResult).filter(TriageResult.patient_id == patient_id,
                                                                TriageResult.visit_id == visit_id).first()
        if results_checker is not None:
            triage_result = TriageResult(
                patient_id=patient_id,
                visit_id=visit_id,
                triage_timestamp=func.now(),
                triage_class=prediction + 1
            )

            db.session.add(triage_result)
        else:
            results_checker.triage_timestamp = func.now()
            results_checker.triage_class = prediction + 1

        db.session.commit()
        return {"message": "Pregnancy status updated successfully", "triageResult": prediction}
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


@app.route("/update_features", methods=["POST"])
def update_features():
    data = request.get_json()
    get_patient_id = data.get("patient_id")
    get_visit_id = data.get("visit_id")

    medical_condition = db.session.query(MedicalCondition).filter(MedicalCondition.patient_id == get_patient_id,
                                                                  MedicalCondition.visit_id == get_visit_id).first()
    historical_abdominal_illness = db.session.query(HistoricalAbdominalIllness).filter(
        HistoricalAbdominalIllness.patient_id == get_patient_id,
        HistoricalAbdominalIllness.visit_id == get_visit_id).first()
    symptom = db.session.query(Symptom).filter(Symptom.patient_id == get_patient_id,
                                               Symptom.visit_id == get_visit_id).first()

    female_patient_info = db.session.query(FemalePatientInfo).filter(FemalePatientInfo.patient_id == get_patient_id,
                                                                     FemalePatientInfo.visit_id == get_visit_id).first()
    if female_patient_info is None:
        female_patient_info.last_menstrual_cycle = 0
        female_patient_info.pregnant_yes = 0
        female_patient_info.pregnant_no = 0
        female_patient_info.pregnant_unsure = 0
        female_patient_info.has_vagina_bleeding = 0
    else:
        female_patient_info.last_menstrual_cycle = data.get("last_menstrual_cycle")
        female_patient_info.pregnant_yes = data.get("pregnant_yes")
        female_patient_info.pregnant_no = data.get("pregnant_no")
        female_patient_info.pregnant_unsure = data.get("pregnant_unsure")
        female_patient_info.has_vagina_bleeding = data.get("has_vagina_bleeding")

    remaining_question = db.session.query(RemainingQuestion).filter(RemainingQuestion.patient_id == get_patient_id,
                                                                    RemainingQuestion.visit_id == get_visit_id).first()

    medical_condition.has_allergy = data.get("has_allergy")
    medical_condition.has_diabetes = data.get("diabetes mellitus")
    medical_condition.has_hyperlipidemia = data.get("hyperlipidemia")
    medical_condition.has_hypertension = data.get("hypertension")
    historical_abdominal_illness.has_abdominal_cancer = data.get("has_abdominal_cancer")
    historical_abdominal_illness.has_chronic_kidney_disease = data.get("chronic kidney disease")
    historical_abdominal_illness.has_gallstone = data.get("gallstone")
    historical_abdominal_illness.has_gastric_problem = data.get("gastric problem")
    historical_abdominal_illness.has_gastroesophageal_reflux_disease = data.get("gastroesophageal reflux disease")
    historical_abdominal_illness.has_irritable_bowel_syndrome = data.get("irritable bowel syndrome")
    historical_abdominal_illness.has_kidney_stones = data.get("has_kidney_stones")
    historical_abdominal_illness.has_liver_disease = data.get("liver disease")
    symptom.has_fever = data.get("fever")
    symptom.has_nausea = data.get("nausea")
    symptom.has_vomiting = data.get("vomit")
    symptom.has_diarrhoea = data.get("diarrhea")
    symptom.has_dysuria = data.get("dysuria")
    symptom.has_abdominal_bloating = data.get("abdominal bloatedness")
    symptom.has_back_pain = data.get("back pain")
    symptom.has_chest_pain = data.get("chest pain")
    symptom.has_constipation = data.get("constipation")
    symptom.has_difficulty_urinating = data.get("difficulty urinating")
    symptom.has_fatigue = data.get("has_fatigue")
    symptom.has_gastric_pain = data.get("gastric pain")
    symptom.has_gastritis = data.get("gastritis")
    symptom.has_heartburn = data.get("has_heartburn")
    symptom.has_appetite_loss = data.get("loss of appetite")
    symptom.has_shivers = data.get("shivers")
    symptom.has_weight_loss = data.get("loss of weight")
    symptom.has_jaundice = data.get("jaundice")
    remaining_question.has_past_abdominal_surgery = data.get("has_past_abdominal_surgery")
    remaining_question.blood_thinning_medication = data.get("consumes blood thinning med")
    remaining_question.is_alcohol_drinker = data.get("is_alcohol_drinker")
    remaining_question.is_smoker = data.get("is_smoker")
    remaining_question.pain_location = data.get("pain_location")
    remaining_question.nature_of_pain = data.get("nature_of_pain")
    remaining_question.colour_of_stool = data.get("stool_colour")
    remaining_question.contents_of_vomit = data.get("contents_of_vomit")
    remaining_question.has_blood_in_stool = data.get("has_bloody_stool")
    remaining_question.has_loose_stool = data.get("has_loose_stool")
    remaining_question.had_bbq_steamboat_rawFood_spicyFood_overnightLeftovers = data.get(
        "has_barbeque_steamboat_raw_food_spicy_food")
    remaining_question.pain_after_meal = data.get("pain_after_meal")

    try:
        db.session.commit()

        prediction = predict_triage(get_patient_id, get_visit_id)

        triage_result = TriageResult(
            patient_id=get_patient_id,
            visit_id=get_visit_id,
            triage_timestamp=func.now(),
            triage_class=prediction + 1
        )
        db.session.add(triage_result)
        db.session.commit()
        return {"message": "Features and triage result updated successfully", "triageResult": prediction}

    except Exception as e:
        print(str(e))
        return {"error": str(e)}, 500


@app.route("/get_all_features/<string:patient_id>/<int:visit_id>", methods=["GET"])
def get_all_features(patient_id, visit_id):
    medical_condition = db.session.query(MedicalCondition).filter(MedicalCondition.patient_id == patient_id,
                                                                  MedicalCondition.visit_id == visit_id).first()
    historical_abdominal_illness = db.session.query(HistoricalAbdominalIllness).filter(
        HistoricalAbdominalIllness.patient_id == patient_id, HistoricalAbdominalIllness.visit_id == visit_id).first()
    symptom = db.session.query(Symptom).filter(Symptom.patient_id == patient_id, Symptom.visit_id == visit_id).first()
    remaining_question = db.session.query(RemainingQuestion).filter(RemainingQuestion.patient_id == patient_id,
                                                                    RemainingQuestion.visit_id == visit_id).first()

    patient_gender = db.session.query(Patient.gender).filter(Patient.patient_id == patient_id).first()
    if patient_gender[0] == "f":
        female_patient_info = db.session.query(FemalePatientInfo).filter(FemalePatientInfo.patient_id == patient_id,
                                                                         FemalePatientInfo.visit_id == visit_id).first()
    else:
        female_patient_info = FemalePatientInfo(
            patient_id=patient_id,
            visit_id=visit_id,
            last_menstrual_cycle=0,
            pregnant_yes=0,
            pregnant_no=0,
            pregnant_unsure=0,
            has_vagina_bleeding=0
        )

    features_info = {
        "medicalCondition": {
            "allergy": medical_condition.has_allergy,
            "diabetesMellitus": medical_condition.has_diabetes,
            "hyperlipidemia": medical_condition.has_hyperlipidemia,
            "hypertension": medical_condition.has_hypertension
        },
        "historicalAbdominalIllness": {
            "abdominalCancer": historical_abdominal_illness.has_abdominal_cancer,
            "chronicKidneyDisease": historical_abdominal_illness.has_chronic_kidney_disease,
            "gallstone": historical_abdominal_illness.has_gallstone,
            "gastricProblem": historical_abdominal_illness.has_gastric_problem,
            "gastroesophagealRefluxDisease": historical_abdominal_illness.has_gastroesophageal_reflux_disease,
            "irritableBowelSyndrome": historical_abdominal_illness.has_irritable_bowel_syndrome,
            "kidneyStones": historical_abdominal_illness.has_kidney_stones,
            "liverDisease": historical_abdominal_illness.has_liver_disease
        },
        "symptom": {
            "fever": symptom.has_fever,
            "nausea": symptom.has_nausea,
            "vomiting": symptom.has_vomiting,
            "diarrhoea": symptom.has_diarrhoea,
            "dysuria": symptom.has_dysuria,
            "abdominalBloating": symptom.has_abdominal_bloating,
            "backPain": symptom.has_back_pain,
            "chestPain": symptom.has_chest_pain,
            "constipation": symptom.has_constipation,
            "difficultyUrinating": symptom.has_difficulty_urinating,
            "fatigue": symptom.has_fatigue,
            "gastricPain": symptom.has_gastric_pain,
            "gastritis": symptom.has_gastritis,
            "heartburn": symptom.has_heartburn,
            "appetiteLoss": symptom.has_appetite_loss,
            "shivers": symptom.has_shivers,
            "weightLoss": symptom.has_weight_loss,
            "jaundice": symptom.has_jaundice
        },
        "femalePatientInfo": {
            "lastMenstrualCycle": female_patient_info.last_menstrual_cycle,
            "pregnant_yes": female_patient_info.pregnant_yes,
            "pregnant_no": female_patient_info.pregnant_no,
            "pregnant_unsure": female_patient_info.pregnant_unsure,
            "vaginaBleeding": female_patient_info.has_vagina_bleeding
        },
        "remainingQuestion": {
            "pastAbdominalSurgery": remaining_question.has_past_abdominal_surgery,
            "bloodThinningMedication": remaining_question.blood_thinning_medication,
            "isAlcoholDrinker": remaining_question.is_alcohol_drinker,
            "isSmoker": remaining_question.is_smoker,
            "painLocation": remaining_question.pain_location,
            "natureOfPain": remaining_question.nature_of_pain,
            "colourOfStool": remaining_question.colour_of_stool,
            "contentsOfVomit": remaining_question.contents_of_vomit,
            "bloodInStool": remaining_question.has_blood_in_stool,
            "looseStool": remaining_question.has_loose_stool,
            "bbqSteamboatRawFoodSpicyFood": remaining_question.had_bbq_steamboat_rawFood_spicyFood_overnightLeftovers,
            "painAfterMeal": remaining_question.pain_after_meal
        }
    }

    return features_info


@app.route("/submit_triage_result", methods=["POST"])
def submit_triage_result():
    data = request.get_json()
    patient = db.session.query(Patient).filter(Patient.patient_id == data.get("patient_id")).first()

    if patient is None:
        return {"error": "Patient not found for the given patient ID"}, 404

    result = db.session.query(TriageResult).filter(TriageResult.patient_id == data.get("patient_id"),
                                                   TriageResult.visit_id == data.get("visit_id")).first()

    if result is None:
        triage_result = TriageResult(
            patient_id=data.get("patient_id"),
            visit_id=data.get("visit_id"),
            triage_timestamp=func.now(),
            triage_class=data.get("triage_class")
        )
        db.session.add(triage_result)
    else:
        result.triage_timestamp = func.now()
        result.triage_class = data.get("triage_class")

    try:
        db.session.commit()
        return {"message": "Triage result submitted successfully"}
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


@app.route("/update_triage_result", methods=["POST"])
def update_triage_result():
    data = request.get_json()

    triage_result = TriageResult(
        patient_id=data.get("patient_id"),
        visit_id=data.get("visit_id"),
        triage_timestamp=func.now(),
        triage_class=data.get("triage_class")
    )

    try:
        db.session.add(triage_result)
        db.session.commit()
        return {"message": "Triage result updated successfully"}
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}


@app.route("/get_triage_result/<string:patient_id>/<int:visit_id>", methods=["GET"])
def get_triage_result(patient_id, visit_id):
    triage_result = db.session.query(TriageResult).filter(TriageResult.patient_id == patient_id,
                                                          TriageResult.visit_id == visit_id).first()
    if triage_result is None:
        return {"error": "Triage result not found for the given patient ID and visit ID"}, 404

    triage_result_for_patient = {
        "patient_id": triage_result.patient_id,
        "visit_id": triage_result.visit_id,
        "triage_timestamp": triage_result.triage_timestamp,
        "triage_class": triage_result.triage_class
    }

    return {"triageResult": triage_result_for_patient.get("triage_class")}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
