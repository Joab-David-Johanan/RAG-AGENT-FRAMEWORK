"use client"

// react hooks for state, refs and effects
import { useState, useRef, useEffect } from "react"

// chat bubble component
import ChatMessage from "@/components/chat-message"

// ui components
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

// sidebar icons
import {
  GitBranch,
  Wrench,
  Users,
  Paperclip,
  X,
  PanelLeftClose,
  PanelLeftOpen,
  Image,
  Database
} from "lucide-react"

export default function Home() {

  // chat history
  // system role added for centered blue informational messages
  const [messages, setMessages] = useState<
    { role: "user" | "assistant" | "system"; content: string }[]
  >([])

  // list of uploaded files shown in sidebar
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])

  // chat input text
  const [input, setInput] = useState("")

  // sidebar open/close state
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // selected rag mode
  const [selectedRag, setSelectedRag] = useState<string | null>(null)

  // hidden file input reference
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // reference to bottom of chat for auto scroll
  const bottomRef = useRef<HTMLDivElement | null>(null)

  // whenever messages change scroll to newest one
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // toggles rag button selection
  function toggleRag(mode: string) {

    // if user clicks the same rag again disable it
    if (selectedRag === mode) {

      setSelectedRag(null)

      // system message when rag is disabled
      setMessages(prev => [
        ...prev,
        { role: "system", content: `${mode} rag disabled.` }
      ])

    } else {

      // activate rag
      setSelectedRag(mode)

      // system message confirming rag selection
      setMessages(prev => [
        ...prev,
        { role: "system", content: `${mode} RAG selected.` }
      ])

    }
  }

  // handles file upload and sends file to backend
  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {

    const file = e.target.files?.[0]
    if (!file) return

    // add filename to sidebar
    setUploadedFiles(prev => [...prev, file.name])

    const formData = new FormData()
    formData.append("file", file)

    try {

      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
      })

      const data = await response.json()

      // show backend message in chat
      setMessages(prev => [
        ...prev,
        { role: "general", content: data.message }
      ])

    } catch {

      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Error processing file." }
      ])

    }
  }

  // remove uploaded file from sidebar list
  function removeFile(index: number) {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  // sends user query to backend rag endpoint
  async function sendMessage() {

    if (!input.trim()) return

    // prevent sending if rag architecture not selected
    if (!selectedRag) {

      // system warning message
      setMessages(prev => [
        ...prev,
        {
          role: "system",
          content: "Select a RAG architecture from the sidebar to continue."
        }
      ])

      return
    }

    const userMessage = input

    // show user message + temporary thinking message
    setMessages(prev => [
      ...prev,
      { role: "user", content: userMessage },
      { role: "assistant", content: "Thinking..." }
    ])

    setInput("")

    try {

      // choose backend endpoint based on selected rag
      let endpoint = ""

      if (selectedRag === "Multi-Agent") {
        endpoint = "http://localhost:8000/chat"
      }

      if (selectedRag === "Multi-Modal") {
        endpoint = "http://localhost:8000/multimodal"
      }

      if (selectedRag === "Adaptive") {
        endpoint = "http://localhost:8000/adaptive"
      }

      if (selectedRag === "Corrective") {
        endpoint = "http://localhost:8000/corrective"
      }

      if (selectedRag === "Cache") {
        endpoint = "http://localhost:8000/cache"
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: userMessage
        })
      })

      const data = await response.json()

      // replace thinking message with real answer
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: data.response }
      ])

    } catch {

      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "Error getting response." }
      ])

    }
  }

  // sidebar button styles
  const buttonBase =
    "w-full justify-start text-2xl py-8 px-6 transition-all duration-200 flex items-center gap-6"

  const hoverStyle =
    "bg-white/10 text-white hover:bg-white/40 hover:text-black hover:scale-[1.02]"

  const activeStyle =
    "bg-white text-black scale-[1.02]"

  return (
    <div className="flex h-screen w-full overflow-hidden">

      {/* sidebar */}

      <aside
        className={`border-r border-white/10 bg-gradient-to-b from-indigo-600 via-purple-600 to-blue-600 p-6 flex flex-col gap-6 transition-all duration-300 ${
          sidebarOpen ? "w-[440px]" : "w-20"
        }`}
      >

        {/* sidebar collapse button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="text-white/80 hover:text-white flex justify-end"
        >
          {sidebarOpen ? <PanelLeftClose size={34} /> : <PanelLeftOpen size={34} />}
        </button>

        {/* sidebar title */}
        {sidebarOpen && (
          <h2 className="text-4xl font-bold text-white mb-4">
            RAG Agents
          </h2>
        )}

        {/* rag mode buttons */}

        <Button
          onClick={() => toggleRag("Adaptive")}
          className={`${buttonBase} ${
            selectedRag === "Adaptive" ? activeStyle : hoverStyle
          }`}
        >
          <GitBranch size={52} />
          {sidebarOpen && "Adaptive RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("Corrective")}
          className={`${buttonBase} ${
            selectedRag === "Corrective" ? activeStyle : hoverStyle
          }`}
        >
          <Wrench size={52} />
          {sidebarOpen && "Corrective RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("Multi-Agent")}
          className={`${buttonBase} ${
            selectedRag === "Multi-Agent" ? activeStyle : hoverStyle
          }`}
        >
          <Users size={52} />
          {sidebarOpen && "Multi-Agent RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("Multi-Modal")}
          className={`${buttonBase} ${
            selectedRag === "Multi-Modal" ? activeStyle : hoverStyle
          }`}
        >
          <Image size={52} />
          {sidebarOpen && "Multi-Modal RAG"}
        </Button>

        <Button
          onClick={() => toggleRag("Cache")}
          className={`${buttonBase} ${
            selectedRag === "Cache" ? activeStyle : hoverStyle
          }`}
        >
          <Database size={52} />
          {sidebarOpen && "Cache RAG"}
        </Button>

        {/* upload section */}

        <div className="mt-6 flex flex-col gap-4">

          <input
            type="file"
            accept=".txt"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
          />

          <Button
            onClick={() => fileInputRef.current?.click()}
            className="w-full justify-start text-2xl py-7 px-6 bg-white/10 text-white hover:bg-white/40 hover:text-black hover:scale-[1.02] transition-all duration-200 flex items-center gap-6"
          >
            <Paperclip size={50} />
            {sidebarOpen && "Upload File"}
          </Button>

          {/* uploaded files list */}

          {sidebarOpen && (
            <div className="text-lg text-white mt-3 space-y-3">

              {uploadedFiles.length === 0 && (
                <p className="text-white/50">No files uploaded</p>
              )}

              {uploadedFiles.map((file, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-white/15 rounded-xl px-4 py-3"
                >
                  <span>{file}</span>

                  <button
                    onClick={() => removeFile(i)}
                    className="text-white/80 hover:text-red-400"
                  >
                    <X size={26} />
                  </button>

                </div>
              ))}

            </div>
          )}

        </div>

      </aside>

      {/* main chat area */}

      <main className="flex min-w-0 flex-1 flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">

        {/* scrollable chat container */}

        <ScrollArea className="flex-1 overflow-y-auto">

          {messages.length === 0 ? (

            <div className="h-full flex items-center justify-center text-white/70">
              Upload a file to build the vector store.
            </div>

          ) : (

            <div className="mx-auto max-w-6xl space-y-8 px-10 py-8">

              {/* render chat messages */}

              {messages.map((m, i) => (
                <ChatMessage
                  key={i}
                  role={m.role}
                  content={m.content}
                />
              ))}

              {/* invisible element used for auto scroll */}

              <div ref={bottomRef} />

            </div>

          )}

        </ScrollArea>

        {/* chat input */}

        <div className="border-t border-white/10 bg-black/20 backdrop-blur p-6">

          <div className="mx-auto flex max-w-6xl gap-3 items-center px-10">

            <Textarea
              placeholder="Ask something..."
              value={input}
              onChange={(e) => setInput(e.target.value)}

              // enter sends message, shift+enter adds new line
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}

              className="min-h-[60px] border-white/10 bg-white/5 text-white"
            />

            <Button
              onClick={sendMessage}
              className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-lg px-8 py-5 rounded-xl shadow-lg hover:scale-[1.03] transition-all duration-200"
            >
              Send
            </Button>

          </div>

        </div>

      </main>

    </div>
  )
}