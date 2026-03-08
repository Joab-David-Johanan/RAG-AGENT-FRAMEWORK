import { Avatar, AvatarFallback } from "@/components/ui/avatar"

interface Props {
  role: "user" | "assistant"
  content: string
}

export default function ChatMessage({ role, content }: Props) {
  return (
    <div className={`flex gap-3 ${role === "user" ? "justify-end" : ""}`}>
      
      {role === "assistant" && (
        <Avatar>
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}

      <div
        className={`max-w-xl rounded-2xl px-5 py-3 text-sm shadow-lg ${
          role === "user"
            ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white"
            : "bg-white/10 border border-white/10 text-white backdrop-blur"
        }`}
      >
        {content}
      </div>

      {role === "user" && (
        <Avatar>
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}