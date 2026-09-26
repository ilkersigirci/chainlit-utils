// Put dictated speech into the chat input so the user can edit it before
// sending. `cl.send_window_message` delivers the transcript to this window.
// Chainlit 2.12 has no API for its input, so the text goes through the
// textarea's native value setter and an input event, which React reads like
// typing.
window.addEventListener("message", (event) => {
  if (
    event.origin !== window.location.origin ||
    event.data?.type !== "chainlit_utils.dictation"
  ) {
    return
  }
  const input = document.getElementById("chat-input")
  if (!input) return
  const text = input.value ? `${input.value} ${event.data.text}` : event.data.text
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set.call(
    input,
    text,
  )
  input.dispatchEvent(new Event("input", { bubbles: true }))
  input.focus()
})
