"use client";

import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";

type Message = { id: string; role: "assistant" | "user"; content: string; mode?: string };
type StorageStatus = "checking" | "saved" | "error";
type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
};

const STORAGE_KEY = "medora-conversations";
const ACTIVE_SESSION_KEY = "medora-active-session";
const USER_KEY = "medora-user";
const MAX_SAVED_CONVERSATIONS = 30;
const apiBase = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

const NEW_CONVERSATION_TITLE = "New conversation";
const WELCOME_CONTENT = "Hi, I’m your medical AI assistant. Ask me a health question, or tell me to save or retrieve your medical records.";
const LEGACY_WELCOME_CONTENTS = [
  "Hi, I’m Medora. Ask me a health question, or tell me to save or retrieve your medical records.",
  "你好，我是 Medora。你可以向我咨询健康问题，也可以让我保存或查询你的医疗记录。",
];

const INITIAL_CONVERSATION: Conversation = {
  id: "initial-session",
  title: NEW_CONVERSATION_TITLE,
  messages: [{
    id: "initial-welcome",
    role: "assistant",
    content: WELCOME_CONTENT,
  }],
  createdAt: 0,
  updatedAt: 0,
};

function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function welcomeMessage(content = WELCOME_CONTENT): Message {
  return { id: createId("welcome"), role: "assistant", content };
}

function createConversation(): Conversation {
  const now = Date.now();
  return {
    id: createId("session"),
    title: NEW_CONVERSATION_TITLE,
    messages: [welcomeMessage()],
    createdAt: now,
    updatedAt: now,
  };
}

function makeTitle(text: string) {
  const normalized = text.replace(/\s+/g, " ").trim();
  const characters = Array.from(normalized);
  return characters.length > 22 ? `${characters.slice(0, 22).join("")}…` : normalized;
}

function isConversation(value: unknown): value is Conversation {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<Conversation>;
  return typeof item.id === "string" && typeof item.title === "string" && Array.isArray(item.messages);
}

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([INITIAL_CONVERSATION]);
  const [activeId, setActiveId] = useState(INITIAL_CONVERSATION.id);
  const [messages, setMessages] = useState<Message[]>(INITIAL_CONVERSATION.messages);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [userId, setUserId] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [historyReady, setHistoryReady] = useState(false);
  const [storageStatus, setStorageStatus] = useState<StorageStatus>("checking");
  const activeIdRef = useRef(activeId);
  const endRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLElement>(null);
  const settingsButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    /* Browser storage is an external source; hydrate it once after the client mounts. */
    /* eslint-disable react-hooks/set-state-in-effect */
    let savedUser = createId("user");
    let savedConversations: Conversation[] = [];
    let savedActiveId: string | null = null;
    let storageSucceeded = false;
    try {
      savedUser = localStorage.getItem(USER_KEY) || savedUser;
      localStorage.setItem(USER_KEY, savedUser);
      savedActiveId = localStorage.getItem(ACTIVE_SESSION_KEY);

      const storedConversations = localStorage.getItem(STORAGE_KEY);
      if (storedConversations) {
        try {
          const parsed = JSON.parse(storedConversations);
          if (Array.isArray(parsed)) {
            savedConversations = parsed.filter(isConversation).map((conversation) => ({
              ...conversation,
              title: conversation.title === "新对话" ? NEW_CONVERSATION_TITLE : conversation.title,
              messages: conversation.messages.map((message) => (
                LEGACY_WELCOME_CONTENTS.includes(message.content) ? { ...message, content: WELCOME_CONTENT } : message
              )),
            }));
          }
        } catch {
          savedConversations = [];
        }
      }
      storageSucceeded = true;
    } catch {
      storageSucceeded = false;
    }
    setUserId(savedUser);

    if (savedConversations.length > 0) {
      const sorted = savedConversations.sort((a, b) => b.updatedAt - a.updatedAt);
      const active = sorted.find((item) => item.id === savedActiveId) || sorted[0];
      setConversations(sorted);
      setActiveId(active.id);
      activeIdRef.current = active.id;
      setMessages(active.messages);
    } else {
      const firstConversation = createConversation();
      setConversations([firstConversation]);
      setActiveId(firstConversation.id);
      activeIdRef.current = firstConversation.id;
      setMessages(firstConversation.messages);
      if (storageSucceeded) {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify([firstConversation]));
          localStorage.setItem(ACTIVE_SESSION_KEY, firstConversation.id);
        } catch {
          storageSucceeded = false;
        }
      }
    }
    setHistoryReady(true);
    setStorageStatus(storageSucceeded ? "saved" : "error");

    fetch(`${apiBase}/health`)
      .then((response) => setOnline(response.ok))
      .catch(() => setOnline(false));
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  useEffect(() => {
    if (!historyReady) return;
    /* Updating this state reports the result of synchronizing to browser storage. */
    /* eslint-disable react-hooks/set-state-in-effect */
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations.slice(0, MAX_SAVED_CONVERSATIONS)));
      localStorage.setItem(ACTIVE_SESSION_KEY, activeId);
      setStorageStatus("saved");
    } catch {
      setStorageStatus("error");
    }
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [activeId, conversations, historyReady]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!showSettings) return;

    function closeOnOutsideClick(event: PointerEvent) {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (!settingsRef.current?.contains(target) && !settingsButtonRef.current?.contains(target)) {
        setShowSettings(false);
      }
    }

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setShowSettings(false);
        settingsButtonRef.current?.focus();
      }
    }

    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [showSettings]);

  function appendMessage(conversationId: string, message: Message) {
    setConversations((current) => current
      .map((conversation) => {
        if (conversation.id !== conversationId) return conversation;
        return {
          ...conversation,
          title: conversation.title === NEW_CONVERSATION_TITLE && message.role === "user" ? makeTitle(message.content) : conversation.title,
          messages: [...conversation.messages, message],
          updatedAt: Date.now(),
        };
      })
      .sort((a, b) => b.updatedAt - a.updatedAt));

    if (activeIdRef.current === conversationId) {
      setMessages((current) => [...current, message]);
    }
  }

  async function sendMessage(value = query) {
    const text = value.trim();
    if (!text || loading) return;

    const requestSessionId = activeIdRef.current;
    appendMessage(requestSessionId, { id: createId("user"), role: "user", content: text });
    setQuery("");
    setLoading(true);

    try {
      const response = await fetch(`${apiBase}/agent_query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, session_id: requestSessionId, user_id: userId }),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed (${response.status})`);
      }

      const data: { answer: string; mode: string } = await response.json();
      appendMessage(requestSessionId, {
        id: createId("assistant"),
        role: "assistant",
        content: data.answer,
        mode: data.mode,
      });
      setOnline(true);
    } catch (error) {
      setOnline(false);
      appendMessage(requestSessionId, {
        id: createId("error"),
        role: "assistant",
        content: error instanceof Error ? `Unable to reach the medical assistant: ${error.message}` : "Unable to reach the medical assistant. Please try again.",
        mode: "error",
      });
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    sendMessage();
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function switchConversation(conversation: Conversation) {
    if (loading || conversation.id === activeIdRef.current) return;
    activeIdRef.current = conversation.id;
    setActiveId(conversation.id);
    setMessages(conversation.messages);
    setQuery("");
    setSidebarOpen(false);
    setShowSettings(false);
  }

  function startConversation() {
    if (loading) return;
    const current = conversations.find((item) => item.id === activeIdRef.current);
    if (current && current.title === NEW_CONVERSATION_TITLE && current.messages.length === 1) {
      setSidebarOpen(false);
      setShowSettings(false);
      return;
    }

    const next = createConversation();
    activeIdRef.current = next.id;
    setConversations((currentItems) => [next, ...currentItems].slice(0, MAX_SAVED_CONVERSATIONS));
    setActiveId(next.id);
    setMessages(next.messages);
    setQuery("");
    setSidebarOpen(false);
    setShowSettings(false);
  }

  function deleteConversation(conversation: Conversation) {
    if (loading) return;
    const confirmed = window.confirm(`Delete “${conversation.title}”? This conversation cannot be recovered.`);
    if (!confirmed) return;

    let remaining = conversations.filter((item) => item.id !== conversation.id);
    if (remaining.length === 0) remaining = [createConversation()];

    setConversations(remaining);
    if (conversation.id === activeIdRef.current) {
      const next = remaining[0];
      activeIdRef.current = next.id;
      setActiveId(next.id);
      setMessages(next.messages);
      setQuery("");
      setShowSettings(false);
      setSidebarOpen(false);
    }
  }

  const activeConversation = conversations.find((item) => item.id === activeId) || conversations[0];
  const isNewConversation = messages.every((message) => message.role !== "user");

  return (
    <main className="app-shell">
      <aside className={`history-sidebar ${sidebarOpen ? "open" : ""}`} aria-label="Recent conversations">
        <div className="sidebar-brand-row">
          <button className="brand" onClick={startConversation} aria-label="Start a new conversation">
            <span className="brand-mark" aria-hidden="true">M</span>
            <span className="brand-subtitle">MEDICAL AI ASSISTANT</span>
          </button>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close recent conversations">×</button>
        </div>

        <button className="new-chat" onClick={startConversation} disabled={loading}>
          <span aria-hidden="true">＋</span> New conversation
        </button>

        <div className="recents">
          <h2>Recents</h2>
          <nav>
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`recent-item ${conversation.id === activeId ? "active" : ""}`}
              >
                <button
                  className="recent-select"
                  onClick={() => switchConversation(conversation)}
                  disabled={loading}
                  title={conversation.title}
                >
                  <span>{conversation.title}</span>
                </button>
                <button
                  className="recent-delete"
                  onClick={() => deleteConversation(conversation)}
                  disabled={loading}
                  aria-label={`Delete ${conversation.title}`}
                  title="Delete conversation"
                >
                  ×
                </button>
              </div>
            ))}
          </nav>
        </div>

        <div className="sidebar-status">
          <span className={`status ${online === true ? "online" : online === false ? "offline" : "checking"}`}>
            <i />{online === true ? "Service online" : online === false ? "Service offline" : "Checking service"}
          </span>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-overlay" onClick={() => setSidebarOpen(false)} aria-label="Close recent conversations" />}

      <section className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            <button className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open recent conversations">☰</button>
            <span>{activeConversation?.title || NEW_CONVERSATION_TITLE}</span>
          </div>
          <div className="top-actions">
            <span className={`privacy-note storage-${storageStatus}`} role="status" aria-live="polite">
              <i>{storageStatus === "saved" ? "✓" : storageStatus === "error" ? "!" : "…"}</i>
              {storageStatus === "saved"
                ? "Saved on this device"
                : storageStatus === "error"
                  ? "Conversation history is not being saved"
                  : "Checking local storage"}
            </span>
            <button
              ref={settingsButtonRef}
              className="icon-button"
              onClick={() => setShowSettings(!showSettings)}
              aria-label="Conversation settings"
              aria-expanded={showSettings}
              aria-haspopup="dialog"
              aria-controls="session-details"
            >•••</button>
          </div>
        </header>

        {showSettings && (
          <aside ref={settingsRef} id="session-details" className="settings" role="dialog" aria-label="Conversation settings">
            <header className="settings-header">
              <div>
                <strong>Session details</strong>
                <p>Current browser context</p>
              </div>
            </header>
            <dl className="settings-details">
              <div>
                <dt>User ID</dt>
                <dd><code>{userId || "Creating…"}</code></dd>
              </div>
              <div>
                <dt>Session ID</dt>
                <dd><code>{activeId || "Creating…"}</code></dd>
              </div>
            </dl>
          </aside>
        )}

        <section className="conversation" aria-live="polite">
          <div className="conversation-inner">
            {isNewConversation && (
              <div className="intro">
                <h1>What can I help you understand?</h1>
                <p>Clear, evidence-aware answers powered by medical knowledge and intelligent tools.</p>
              </div>
            )}

            <div className="messages">
              {messages.map((message) => (
                <article key={message.id} className={`message ${message.role} ${message.mode === "error" ? "message-error" : ""}`}>
                  {message.role === "assistant" && <div className="avatar" aria-hidden="true">M</div>}
                  <div className="message-body">
                    <div className="message-meta">
                      <strong>{message.role === "assistant" ? "Assistant" : "You"}</strong>
                      {message.mode && message.mode !== "error" && <span>Agent</span>}
                    </div>
                    <p>{message.content}</p>
                  </div>
                </article>
              ))}
              {loading && (
                <article className="message assistant">
                  <div className="avatar" aria-hidden="true">M</div>
                  <div className="message-body thinking"><span /><span /><span /><em>Looking for reliable information</em></div>
                </article>
              )}
              <div ref={endRef} />
            </div>

          </div>
        </section>

        <footer className="composer-wrap">
          <div className="composer-inner">
            <form className="composer" onSubmit={handleSubmit}>
              <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={handleKeyDown} placeholder="Ask a health question…" aria-label="Health question" rows={1} />
              <button type="submit" disabled={!query.trim() || loading} aria-label="Send question">↑</button>
            </form>
            <p className="disclaimer">This assistant provides general health information and does not replace professional diagnosis or treatment. In an emergency, contact your local emergency services immediately.</p>
          </div>
        </footer>
      </section>
    </main>
  );
}
