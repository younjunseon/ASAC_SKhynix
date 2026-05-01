import { useEffect, useRef, useState } from "react";

interface Message {
  role: "user" | "bot";
  text: string;
  ts: number;
}

const SUGGESTIONS = [
  "오늘 위험 wafer 알려줘",
  "최근 불량률 추이는?",
  "어떤 변수가 문제야?",
  "위험 unit 목록",
];

function botReply(input: string): string {
  const text = input.toLowerCase();
  if (text.includes("위험") && text.includes("wafer")) {
    return "Drill-down 탭에서 lot/wafer 트리 상단을 보세요. wafer 선택 시 die 위치와 주요 기여 변수가 함께 표시됩니다.";
  }
  if (text.includes("불량률") || text.includes("추이")) {
    return "Overview 시계열 차트에서 일/주/월 단위 추이를 확인할 수 있습니다. 추세 검정 결과는 차트 우측 상단 칩에 표시됩니다.";
  }
  if (text.includes("변수") || text.includes("문제")) {
    return "Drill-down에서 unit 선택 시 우측 패널에 주요 기여 변수 Top 20과 z-score가 표시됩니다. 전반적인 변수 영향력은 Model 탭에서 확인 가능합니다.";
  }
  if (text.includes("unit")) {
    return "Data 탭에서 status·검색·정렬로 unit을 조회할 수 있습니다.";
  }
  return "준비 중입니다. 이후 데이터 질의와 변수 근거 분석까지 응답하도록 확장 예정입니다.";
}

export default function ChatbotWidget() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      text: "무엇을 도와드릴까요?",
      ts: Date.now(),
    },
  ]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, open]);

  function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;
    const userMsg: Message = { role: "user", text: trimmed, ts: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: botReply(trimmed), ts: Date.now() },
      ]);
    }, 400);
  }

  return (
    <>
      {/* 플로팅 버튼 */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-4 right-4 xl:bottom-6 xl:right-6 w-11 h-11 xl:w-12 xl:h-12 rounded-full text-white flex items-center justify-center transition-all hover:scale-105 z-40"
        style={{
          background: "linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)",
          boxShadow: "0 6px 20px rgba(37, 99, 235, 0.35)",
        }}
        aria-label="챗봇 열기"
      >
        {open ? (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.86 9.86 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        )}
      </button>

      {/* 챗봇 패널 */}
      {open && (
        <div
          className="fixed bottom-20 right-6 w-[360px] h-[520px] bg-white rounded-2xl shadow-cardHover border border-brand-border z-40 flex flex-col overflow-hidden"
          style={{ boxShadow: "0 12px 40px rgba(15, 23, 42, 0.18)" }}
        >
          {/* 헤더 */}
          <div
            className="px-4 py-3 text-white flex items-center gap-3"
            style={{ background: "linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)" }}
          >
            <div className="w-9 h-9 rounded-full bg-white/15 ring-2 ring-white/20 flex items-center justify-center">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-semibold text-[14px]">Assistant</div>
              <div className="text-[11px] text-white/70 flex items-center gap-1">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-brand-success"></span>
                온라인
              </div>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="w-7 h-7 rounded-full hover:bg-white/15 flex items-center justify-center"
              aria-label="닫기"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* 메시지 영역 */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3 bg-brand-subtle">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] px-3 py-2 rounded-2xl text-[12.5px] leading-relaxed ${
                    m.role === "user"
                      ? "bg-brand-primary text-white rounded-br-sm"
                      : "bg-white text-brand-text shadow-card rounded-bl-sm"
                  }`}
                >
                  {m.text}
                </div>
              </div>
            ))}
          </div>

          {/* 빠른 질문 */}
          {messages.length <= 1 && (
            <div className="px-3 py-2 border-t border-brand-border bg-white">
              <div className="text-[10px] text-brand-textMuted mb-1.5 px-1">추천 질문</div>
              <div className="flex flex-wrap gap-1.5">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="text-[11px] px-2.5 py-1 rounded-full border border-brand-border text-brand-textMuted hover:border-brand-primary hover:text-brand-primary transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 입력 */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="px-3 py-2.5 border-t border-brand-border bg-white flex items-center gap-2"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="메시지를 입력하세요…"
              className="flex-1 text-[12.5px] px-3 py-2 rounded-full border border-brand-border focus:outline-none focus:border-brand-primary bg-brand-subtle"
            />
            <button
              type="submit"
              disabled={!input.trim()}
              className="w-9 h-9 rounded-full bg-brand-primary text-white flex items-center justify-center disabled:opacity-40 hover:bg-brand-primaryDark transition-colors"
              aria-label="전송"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            </button>
          </form>
        </div>
      )}
    </>
  );
}
