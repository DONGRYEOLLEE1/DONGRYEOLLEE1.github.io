export const SITE = {
  title: '동렬', description: 'LLM 학습과 추론, AI 에이전트 개발의 실험과 구현을 기록합니다.',
  author: '동렬', github: 'https://github.com/DONGRYEOLLEE1', huggingface: 'https://huggingface.co/drlee1',
  origin: (process.env.SITE_URL || (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : '') || 'https://dongryeollee1.github.io').replace(/\/+$/, ''),
};
export const TYPE_LABELS = { research: '연구', 'paper-review': '논문 리뷰', implementation: '구현기', troubleshooting: '문제 해결', note: '학습 노트' };
export const formatDate = (date: string) => date.slice(0,10).replaceAll('-', '.');
export const longDate = (date: string) => new Intl.DateTimeFormat('ko-KR',{year:'numeric',month:'long',day:'numeric',timeZone:'Asia/Seoul'}).format(new Date(date));
export const absoluteUrl = (path: string) => new URL(path, SITE.origin).href;
