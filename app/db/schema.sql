-- RadiScribe AI Engine - Production Schema
-- Designed for Supabase / PostgreSQL

-- 1. STUDIES (The Images)
create table studies (
  study_id uuid primary key default uuid_generate_v4(),
  image_hash text unique not null, -- SHA256 of file content
  storage_path text not null, -- "ab/cd/hash.png"
  modality text default 'DX',
  created_at timestamptz default now()
);

-- 2. PREDICTIONS (Immutable AI Output)
create table ai_predictions (
  prediction_id uuid primary key default uuid_generate_v4(),
  study_id uuid references studies(study_id),
  model_version text not null,
  findings jsonb not null, -- [{label: "Pneumonia", prob: 0.9, confidence: "HIGH"}]
  heatmap_url text, 
  raw_output jsonb, -- Full inference output for debugging
  created_at timestamptz default now()
);

-- 3. FEEDBACK (Active Learning Loop)
create table feedback (
  feedback_id uuid primary key default uuid_generate_v4(),
  prediction_id uuid references ai_predictions(prediction_id),
  user_id text, -- Radiologist ID
  feedback_type text check (feedback_type in ('ACCEPTED', 'CORRECTED', 'REJECTED')),
  correction_details jsonb, -- {added: ["Normal"], removed: ["Pneumonia"]}
  created_at timestamptz default now()
);

-- 4. AUDIT LOG (Compliance)
create table audit_logs (
  log_id uuid primary key default uuid_generate_v4(),
  action text not null, -- "ANALYZE_REQUEST", "FEEDBACK_SUBMIT"
  actor_id text, -- System or User
  resource_id uuid,
  details jsonb,
  timestamp timestamptz default now()
);

-- INDEXES
create index idx_studies_hash on studies(image_hash);
create index idx_feedback_prediction on feedback(prediction_id);
create index idx_predictions_study on ai_predictions(study_id);

-- RLS (Row Level Security) - Basic Template
alter table studies enable row level security;
alter table ai_predictions enable row level security;
alter table feedback enable row level security;

-- Policy: Service Role (API) has full access
create policy "Service Role Full Access" on studies
  for all using ( auth.role() = 'service_role' );

