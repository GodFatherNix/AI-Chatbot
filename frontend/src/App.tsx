import React from "react";
import ChatWindow from "./components/ChatWindow";

const App: React.FC = () => {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "1rem" }}>
      <ChatWindow />
    </div>
  );
};

export default App;