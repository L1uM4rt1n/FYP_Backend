-- for martin use -- mysql -u root -p < ./database/script.sql (BE dir)
-- for deployment -- mysql -u b5cf0d40805bd4 -h mysql -u b5cf0d40805bd4 -p -h us-cluster-east-01.k8s.cleardb.net heroku_ea7263fb720321f < ./database/script.sql

-- CREATE DATABASE IF NOT EXISTS triagedb;
-- USE triagedb;

DROP TABLE IF EXISTS triage_results;
DROP TABLE IF EXISTS remaining_questions;
DROP TABLE IF EXISTS female_patient_info;
DROP TABLE IF EXISTS symptoms;
DROP TABLE IF EXISTS historical_abdominal_illness;
DROP TABLE IF EXISTS medical_conditions;
DROP TABLE IF EXISTS vitals;
DROP TABLE IF EXISTS visits;
DROP TABLE IF EXISTS patients;

CREATE TABLE patients (
    patient_id VARCHAR(255) PRIMARY KEY,
    age INT,
    gender VARCHAR(255),
    phone_number VARCHAR(255)
);

CREATE TABLE visits (
    visit_id INT,
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    date_of_visit DATE,
    time_of_visit TIME,
    PRIMARY KEY (visit_id, patient_id)
);

CREATE TABLE vitals (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    heart_rate INT,
    diastolic_bp INT,
    systolic_bp INT,
    resp_rate INT,
    pain_score INT,
    spo2 INT,
    temp FLOAT,
    avpu_score VARCHAR(255),
    adjusted_si FLOAT,
    mews_score INT,
    mode_of_arrival VARCHAR(255),
    PRIMARY KEY (patient_id, visit_id)
);

CREATE TABLE medical_conditions (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    has_allergy VARCHAR(255),
    has_diabetes INT,
    has_hyperlipidemia INT,
    has_hypertension INT,
    PRIMARY KEY (patient_id, visit_id)
);

CREATE TABLE historical_abdominal_illness (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    has_abdominal_cancer INT,
    has_chronic_kidney_disease INT,
    has_gallstone INT,
    has_gastric_problem INT,
    has_gastroesophageal_reflux_disease INT,
    has_irritable_bowel_syndrome INT,
    has_kidney_stones INT,
    has_liver_disease INT,
    PRIMARY KEY (patient_id, visit_id)
);

CREATE TABLE symptoms (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    has_fever INT,
    has_nausea INT,
    has_vomiting INT,
    has_diarrhoea INT,
    has_dysuria INT,
    has_abdominal_bloating INT,
    has_back_pain INT,
    has_chest_pain INT,
    has_constipation INT,
    has_difficulty_urinating INT,
    has_fatigue INT,
    has_gastric_pain INT,
    has_gastritis iNT,
    has_heartburn INT,
    has_appetite_loss INT,
    has_shivers INT,
    has_weight_loss INT,
    has_jaundice INT,
    PRIMARY KEY (patient_id, visit_id)
);

CREATE TABLE female_patient_info (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    last_menstrual_cycle INT,
    pregnant_yes INT,
    pregnant_no INT,
    pregnant_unsure INT,
    has_vagina_bleeding INT,
    PRIMARY KEY (patient_id, visit_id)
);

CREATE TABLE remaining_questions (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    has_past_abdominal_surgery INT,
    blood_thinning_medication INT,
    is_alcohol_drinker INT,
    is_smoker INT,
    pain_location VARCHAR(255),
    nature_of_pain VARCHAR(255),
    colour_of_stool VARCHAR(255),
    contents_of_vomit VARCHAR(255),
    has_blood_in_stool INT,
    has_loose_stool INT,
    had_bbq_steamboat_rawFood_spicyFood_overnightLeftovers INT,
    pain_after_meal INT,
    PRIMARY KEY (patient_id, visit_id)
);

CREATE TABLE triage_results (
    patient_id VARCHAR(255),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    visit_id INT,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
    triage_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    triage_class INT,
    PRIMARY KEY (patient_id, visit_id, triage_timestamp)
);
