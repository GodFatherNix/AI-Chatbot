import React, { useState } from "react";

interface Props {
  message: {
    id: string;
    role: "user" | "assistant";
    content: string;
    citations?: { id: string; source?: string }[];
  };
}

const Message: React.FC<Props> = ({ message }) => {
  const { role, content, citations } = message;
  const [showSources, setShowSources] = useState<boolean>(false);

  const isUser = role === "user";

  return (
    <div
      style={{
        textAlign: isUser ? "right" : "left",
        marginBottom: "0.75rem",
      }}
    >
      <div
        style={{
          display: "inline-block",
          background: isUser ? "#0078D4" : "#e5e5ea",
          color: isUser ? "white" : "black",
          padding: "0.5rem 0.75rem",
          borderRadius: 12,
          maxWidth: "80%",
          whiteSpace: "pre-wrap",
        }}
      >
        {content}
        {citations && citations.length > 0 && !isUser && (
          <button
            onClick={() => setShowSources(!showSources)}
            style={{
              marginLeft: 8,
              fontSize: 12,
              background: "transparent",
              border: "none",
              cursor: "pointer",
              color: isUser ? "white" : "#0078D4",
            }}
          >
            [{citations.length}]
          </button>
        )}
      </div>
      {showSources && citations && citations.length > 0 && (
        <ul style={{ listStyle: "disc", paddingLeft: 20, marginTop: 4 }}>
          {citations.map((c) => (
            <li key={c.id} style={{ fontSize: 12 }}>
              {c.source || c.id}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Message;