-- 연관 글 추천의 블록 임베딩 공유 캐시.
-- 두 발행 서버가 같은 벡터로 후보를 계산하도록 (프로필, 입력 해시)로 조회한다.
-- Supabase SQL Editor에서 한 번 실행한다. 차원을 바꾸면 vector(768)도 함께 바꾼다.

create extension if not exists vector;

create table if not exists block_embeddings (
  profile text not null,
  input_hash text not null,
  embedding vector(768) not null,
  created_at timestamptz not null default now(),
  primary key (profile, input_hash)
);

-- 정책을 만들지 않는다: RLS를 우회하는 service role 키만 읽고 쓸 수 있다.
alter table block_embeddings enable row level security;

-- 새 테이블에 API 역할 권한을 자동으로 주지 않는 프로젝트에서도 동작하도록 명시한다.
-- 이미 부여돼 있으면 아무 일도 일어나지 않는다.
grant select, insert, update on table block_embeddings to service_role;
revoke all on table block_embeddings from anon, authenticated;
