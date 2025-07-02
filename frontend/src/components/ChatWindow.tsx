import React, { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import Message from "./Message";

interface BackendResponse {
  answer: string;
  context: string[];
  citations: { id: string; source?: string }[];
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: { id: string; source?: string }[];
}

const ChatWindow: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const sessionId = useRef<string>(uuidv4());
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg: ChatMessage = {
      id: uuidv4(),
      role: "user",
      content: input.trim(),
    };
    setMessages((prev: ChatMessage[]) => [...prev, userMsg]);
    setInput("");

    setLoading(true);
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId.current,
          message: userMsg.content,
          top_k: 5,
        }),
      });
      if (!res.ok) {
        throw new Error(`Backend error: ${await res.text()}`);
      }
      const data = (await res.json()) as BackendResponse;
      const assistantMsg: ChatMessage = {
        id: uuidv4(),
        role: "assistant",
        content: data.answer,
        citations: data.citations,
      };
      setMessages((prev: ChatMessage[]) => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      const errorMsg: ChatMessage = {
        id: uuidv4(),
        role: "assistant",
        content: "Sorry, I couldn't reach the server.",
      };
      setMessages((prev: ChatMessage[]) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div
      style={{
        width: "100%",
        maxWidth: 600,
        border: "1px solid #ddd",
        borderRadius: 8,
        display: "flex",
        flexDirection: "column",
        height: "80vh",
      }}
    >
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "1rem",
          background: "#fafafa",
        }}
      >
        {messages.map((msg: ChatMessage) => (
          <Message key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
      <div style={{ display: "flex", padding: "0.5rem", borderTop: "1px solid #eee" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          style={{ flex: 1, padding: "0.5rem" }}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          style={{ marginLeft: "0.5rem" }}
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;