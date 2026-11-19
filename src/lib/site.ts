export const SITE = {
  title: 'Tech Blog', description: 'LLM 학습과 추론, AI 에이전트 개발의 실험과 구현을 기록합니다.',
  author: '동렬', github: 'https://github.com/DONGRYEOLLEE1', huggingface: 'https://huggingface.co/drlee1',
  origin: (process.env.SITE_URL || (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : '') || 'https://dongryeollee1.github.io').replace(/\/+$/, ''),
};
export const absoluteUrl = (path: string) => new URL(path, SITE.origin).href;
