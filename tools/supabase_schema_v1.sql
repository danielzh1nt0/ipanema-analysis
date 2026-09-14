-- Ipanema v1 schema. Paste into Supabase → SQL editor → Run.
create table if not exists matches (
  id text primary key,
  created_at timestamptz default now(),
  status text not null default 'ready' check (status in ('processing','ready','failed')),
  duration_s real, fps real, width int, height int,
  schema_version int, contract text,
  summary jsonb, attack_right jsonb, attack_right_confidence jsonb,
  files jsonb            -- {video, match_data, stats, kit_A, kit_B, thumb} = storage object paths in bucket "matches"
);
create table if not exists match_labels (
  match_id text primary key references matches(id) on delete cascade,
  club_team text check (club_team in ('A','B')),
  name_a text, name_b text, colour_a text, colour_b text,
  opponent text, score_a int, score_b int, date date, competition text, tags text[],
  attack_right_override jsonb, thresholds jsonb,
  updated_at timestamptz default now()
);
-- storage bucket for match files (private; the app reads via signed URLs)
insert into storage.buckets (id, name, public) values ('matches', 'matches', false) on conflict (id) do nothing;
-- allow the app (anon key) to read rows; writes come from the pipeline with the service key
alter table matches enable row level security;
alter table match_labels enable row level security;
create policy "read matches" on matches for select using (true);
create policy "read labels" on match_labels for select using (true);
create policy "write labels" on match_labels for all using (true) with check (true);
create policy "read match files" on storage.objects for select using (bucket_id = 'matches');
