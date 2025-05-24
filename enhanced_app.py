import streamlit as st
import os
import tempfile
import base64
from PIL import Image
import io
import glob
import google.generativeai as genai # For GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from streamlit_extras.stylable_container import stylable_container
import time
import fitz  # PyMuPDF
# Add these imports for video processing
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import asyncio
import threading
import queue
import numpy as np
import pyaudio
import json
from google.genai import types as google_genai_types
from google import genai as live_genai_module # For Gemini Live Client

# Audio constants for Gemini Live
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"

# Set page config
st.set_page_config(page_title="ESAB Welding Assistant", layout="wide", page_icon="🔧")

# Add custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #181818;
        color: #f5f5f5;
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #FF9800;
        margin-bottom: 20px;
        text-align: center;
    }
    .feature-header {
        font-size: 24px;
        font-weight: bold;
        color: #FFB74D;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #263238;
        border-left: 5px solid #29B6F6;
        color: #f5f5f5;
    }
    .assistant-message {
        background-color: #212121;
        border-left: 5px solid #FF9800;
        color: #f5f5f5;
    }
    .stButton>button {
        background-color: #FF9800;
        color: #181818;
        font-weight: bold;
    }
    .info-box {
        background-color: #333333;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        color: #f5f5f5;
    }
    .stProgress > div > div {
        background-color: #FF9800;
    }
    .salesman-mode {
        background-color: #2E7D32;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #81C784;
        color: #f5f5f5;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API
def init_genai():
    try:
        # You should store your API key in Streamlit secrets or environment variables
        api_key = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please add it to your Streamlit secrets or environment variables.")
            st.stop()
        
        genai.configure(api_key=api_key)
        
        # Configure safety settings for the model
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Generation config
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        # Create the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Create a flash model for streaming
        flash_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model, flash_model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        st.stop()

# Load the manuals
def load_manuals():
    """Find and load all available welding machine manuals"""
    manual_files = glob.glob("manuals/*.pdf")
    manuals = {}
    
    for manual_path in manual_files:
        filename = os.path.basename(manual_path)
        # Extract machine name from filename - can be customized based on actual naming convention
        machine_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
        manuals[machine_name] = manual_path
    
    return manuals

# Extract text from a PDF
def extract_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Extract images from a PDF
def extract_pdf_images(pdf_path):
    """Extract images from a PDF file"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert image bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append({
                    "page": page_num + 1,
                    "image": image,
                    "bytes": image_bytes
                })
                
    except Exception as e:
        st.error(f"Error extracting images from PDF: {e}")
    
    return images

# Create a system prompt for the selected machine
def create_system_prompt(machine_name, manual_path):
    """Create a system prompt with the manual content"""
    manual_text = extract_pdf_text(manual_path)
    
    system_prompt = f"""
    You are an expert ESAB welding machine assistant specializing in the {machine_name} model.
    You have been trained on the manual for this machine and can provide accurate information
    about error codes, component identification, troubleshooting, and repair procedures.
    
    Here is the manual content to reference:
    {manual_text[:100000]}  # We limit to first 100K chars as prompt context
    
    When answering questions:
    1. Always base your answers on the manual content.
    2. If asked about error codes, provide the code meaning and recommended actions.
    3. If asked about components, provide the part name, part number, and location if available.
    4. For troubleshooting, give step-by-step guidance with safety precautions.
    5. Always prioritize safety in your recommendations.
    6. If asked about something not in the manual, admit that you don't have that specific information.
    
    Format your responses clearly with headings, bullet points, and numbered steps where appropriate.
    """
    
    return system_prompt

# Function to analyze uploaded component image
def analyze_component_image(image, machine_name, manual_path, model):
    """Analyze an uploaded image of a component and identify it"""
    # Extract images from the manual for reference
    manual_images = extract_pdf_images(manual_path)
    manual_text = extract_pdf_text(manual_path)
    
    # Convert the uploaded image to bytes for the model
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create a prompt for the model
    prompt = f"""
    I have uploaded an image of a component from an ESAB {machine_name} welding machine.
    Please identify this component by:
    1. The name of the component
    2. The part number if visible or if you can determine it
    3. Its function in the welding machine
    4. Which board or assembly it belongs to
    
    Use the manual content to assist in identification:
    {manual_text[:50000]}
    
    If you cannot identify the exact component, suggest the closest matches and explain why.
    """
    
    # Send the image and prompt to the model
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_str}
        ])
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {e}"

# Function to process fault list and recommend spare parts
def process_fault_list(fault_pdf, machine_name, manual_path, model):
    """Process a fault list PDF and recommend spare parts"""
    # Extract text from the uploaded fault PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(fault_pdf.getvalue())
        tmp_path = tmp.name
    
    fault_text = extract_pdf_text(tmp_path)
    manual_text = extract_pdf_text(manual_path)
    
    # Clean up the temp file
    os.unlink(tmp_path)
    
    # Create a prompt for the model
    prompt = f"""
    I have a fault list for an ESAB {machine_name} welding machine.
    The fault list contains the following information:
    
    {fault_text}
    
    Based on the manual for this machine:
    {manual_text[:50000]}
    
    Please:
    1. Extract and list all fault codes from the uploaded document
    2. For each fault code, identify the most likely components that could cause this fault
    3. Rank the top 3 spare parts recommended for inspection or replacement
    4. Provide the part numbers for these components if available
    5. Suggest a troubleshooting approach for each fault
    
    Format your response as a structured report with sections for each fault code.
    """
    
    # Send the prompt to the model
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error processing fault list: {e}"

# Function to generate a repair playbook
def generate_repair_playbook(issue, machine_name, manual_path, model):
    """Generate a step-by-step repair playbook for the selected issue"""
    manual_text = extract_pdf_text(manual_path)
    
    # Create a prompt for the model
    prompt = f"""
    Create a detailed repair playbook for fixing the following issue on an ESAB {machine_name} welding machine:
    
    Issue: {issue}
    
    Using the manual content:
    {manual_text[:50000]}
    
    Generate a comprehensive, step-by-step repair guide that includes:
    
    1. Required tools and safety equipment
    2. Safety precautions and warnings specific to this repair
    3. Preparation steps (power off, disconnect, etc.)
    4. Detailed numbered steps for the repair process with clear instructions
    5. Testing procedures to verify the repair was successful
    6. Reassembly instructions if needed
    7. Additional tips or common mistakes to avoid
    
    Format the playbook with clear headings, numbered steps, and emphasize all safety warnings.
    Reference specific parts by their correct names and part numbers whenever possible.
    Include any relevant troubleshooting steps that should be performed before component replacement.
    """
    
    # Send the prompt to the model
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating repair playbook: {e}"

# Function to handle streaming responses
def stream_response(prompt, flash_model):
    """Stream a response from the model"""
    try:
        response = flash_model.generate_content(prompt, stream=True)
        
        # Create a placeholder for the streamed response
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
        # Show the final response without the cursor
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        return full_response
    
    except Exception as e:
        st.error(f"Error streaming response: {e}")
        return None

# Create a salesman mode prompt
def create_salesman_prompt(machine_name, manual_path):
    """Create a prompt for the salesman mode"""
    manual_text = extract_pdf_text(manual_path)
    
    salesman_prompt = f"""Speak as salesman as natural as possible.
    You are an expert ESAB welding machine sales representative specializing in the {machine_name} model.
    You will be interacting with a potential customer who is interested in this welding machine.
    
    Here is technical information about the machine from its manual:
    {manual_text[:50000]}
    
    Your job is to:
    1. Enthusiastically highlight the key features and benefits of the {machine_name}
    2. Emphasize its unique selling points compared to competitors
    3. Explain technical specifications in simple terms
    4. Address any concerns or questions about the machine
    5. Guide the customer through the purchase decision process
    6. Suggest accessories or add-ons that would enhance their experience
    
    Speak in a professional, friendly, and persuasive manner.
    Use sales techniques like benefit highlighting, social proof, and mild urgency where appropriate.
    Never be pushy or make false claims about the product.
    If you don't know something specific, acknowledge it and offer to connect them with technical support.
    
    Remember, your goal is to help the customer understand why this machine is the right choice for their welding needs.
    """
    
    return salesman_prompt

# Video processor class for the webcam stream
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_screenshot = None
        self.take_screenshot = False
    
    def toggle_screenshot(self):
        self.take_screenshot = True
    
    def get_screenshot(self):
        return self.last_screenshot
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Take screenshot when requested
        if self.take_screenshot:
            self.last_screenshot = img.copy()
            self.take_screenshot = False
        
        # Add frame counter
        self.frame_count += 1
        
        # Optional: Add text overlay to show it's live
        cv2.putText(
            img,
            f"ESAB Live Feed - Frame: {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),  # Orange color
            2,
        )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Gemini Live Audio Processing
class GeminiLiveAudio:
    def __init__(self, machine_name, manual_path):
        self.machine_name = machine_name
        self.manual_path = manual_path
        self.running = False
        self.conversation_history = []
        
        # Initialize PyAudio
        self.pya = pyaudio.PyAudio()
        
        # Audio streams
        self.audio_stream = None
        self.output_stream = None
          # Async components
        self.audio_in_queue = None
        self.out_queue = None
        self.text_queue = None  # Queue for text messages
        self.session = None
        self.current_loop = None
        self.tasks = []
        
        # Generate the salesman prompt
        self.system_prompt = create_salesman_prompt(machine_name, manual_path)
        
        # Initialize Gemini Live client
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found for Gemini Live")
            return # Ensure we don't proceed without an API key
            
        self.client = live_genai_module.Client( # Use live_genai_module here
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )
        
        # Configure Gemini Live settings
        self.config = google_genai_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=google_genai_types.SpeechConfig(
                voice_config=google_genai_types.VoiceConfig(
                    prebuilt_voice_config=google_genai_types.PrebuiltVoiceConfig(voice_name="Charan")
                )
            ),
            context_window_compression=google_genai_types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=google_genai_types.SlidingWindow(target_tokens=12800),
            ),
        )
    
    async def listen_audio(self):
        """Capture audio from microphone and send to Gemini Live"""
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        while self.running:
            try:
                data = await asyncio.to_thread(
                    self.audio_stream.read, 
                    CHUNK_SIZE, 
                    exception_on_overflow=False
                )
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                st.error(f"Error capturing audio: {e}")
                break
    async def send_realtime(self):
        """Send audio data to Gemini Live"""
        while self.running:
            try:
                msg = await self.out_queue.get()
                await self.session.send(input=msg)
            except Exception as e:
                st.error(f"Error sending audio: {e}")
                break
    
    async def send_text_messages(self):
        """Send text messages from the text queue to Gemini Live"""
        while self.running:
            try:
                text_msg = await self.text_queue.get()
                await self.session.send(input=text_msg, end_of_turn=True)
            except Exception as e:
                st.error(f"Error sending text: {e}")
                break
    
    async def receive_audio(self):
        """Receive audio responses from Gemini Live"""
        while self.running:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        # Add text response to conversation history
                        self.conversation_history.append({
                            "role": "assistant", 
                            "content": text
                        })
                
                # Clear audio queue on turn complete for interruptions
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
                    
            except Exception as e:
                st.error(f"Error receiving audio: {e}")
                break
    
    async def play_audio(self):
        """Play audio responses from Gemini Live"""
        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        while self.running:
            try:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(self.output_stream.write, bytestream)
            except Exception as e:
                st.error(f"Error playing audio: {e}")
                break
    async def send_initial_prompt(self):
        """Send the initial system prompt to set up the salesman context"""
        await self.session.send(input=self.system_prompt, end_of_turn=True)
        
    async def run_session(self):
        """Main async session runner"""
        try:
            async with (
                self.client.aio.live.connect(model=MODEL, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.text_queue = asyncio.Queue()  # Initialize text queue
                
                # Send initial salesman prompt
                await self.send_initial_prompt()
                
                # Create tasks
                self.tasks = [
                    tg.create_task(self.send_realtime()),
                    tg.create_task(self.send_text_messages()),  # Add text message task
                    tg.create_task(self.listen_audio()),
                    tg.create_task(self.receive_audio()),
                    tg.create_task(self.play_audio())
                ]
                
                # Wait for stop signal
                while self.running:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            st.error(f"Error in Gemini Live session: {e}")
        finally:
            self.cleanup()
    
    def start(self):
        """Start the Gemini Live session"""
        if not self.running:
            self.running = True
            
            # Run the async session in a separate thread
            def run_async_session():
                try:
                    asyncio.run(self.run_session())
                except Exception as e:
                    st.error(f"Error starting async session: {e}")
            
            import threading
            self.session_thread = threading.Thread(target=run_async_session, daemon=True)
            self.session_thread.start()
            
            return True
        return False
    
    def stop(self):
        """Stop the Gemini Live session"""
        if self.running:
            self.running = False
            self.cleanup()
            return True
        return False
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
    
    async def send_text_input_async(self, text):
        """Send text input to Gemini Live"""
        if self.session:
            await self.session.send(input=text, end_of_turn=True)
            self.conversation_history.append({"role": "user", "content": text})
    def send_text_input(self, text):
        """Send text input (sync wrapper)"""
        if self.running and self.session and self.text_queue:
            # Add text to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Put text message in queue to be sent by async task
            try:
                self.text_queue.put_nowait(text)
                return f"Text message sent: {text}"
            except Exception as e:
                return f"Error queuing text message: {e}"
        else:
            return "Gemini Live session not active"

# Main app
def main():
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'selected_machine' not in st.session_state:
        st.session_state.selected_machine = None
        
    if 'model' not in st.session_state or 'flash_model' not in st.session_state:
        model, flash_model = init_genai()
        st.session_state.model = model
        st.session_state.flash_model = flash_model
    
    if 'gemini_live_audio' not in st.session_state:
        st.session_state.gemini_live_audio = None
    
    # Add a callback function to clear the chat input
    def clear_chat_input():
        st.session_state.chat_input = ""
    
    # Load available manuals
    manuals = load_manuals()
    
    # Header
    st.markdown('<div class="main-header">ESAB Welding Machine Assistant</div>', unsafe_allow_html=True)
    
    # Display ESAB logo or a welding image here if desired
    # st.image("esab_logo.png", width=200)
    
    # 1. Machine Selector
    st.markdown('<div class="feature-header">1. Select Your Welding Machine</div>', unsafe_allow_html=True)
    
    if not manuals:
        st.warning("No machine manuals found. Please add PDF manuals to the 'manuals' folder.")
    else:
        machine_options = list(manuals.keys())
        selected_machine = st.selectbox(
            "Choose your ESAB welding machine model",
            options=machine_options,
            index=0 if machine_options else None
        )
        
        if selected_machine:
            st.session_state.selected_machine = selected_machine
            manual_path = manuals[selected_machine]
            st.session_state.manual_path = manual_path
            
            st.markdown(f'<div class="info-box">Selected machine: <b>{selected_machine}</b><br>Manual: {os.path.basename(manual_path)}</div>', unsafe_allow_html=True)
            
            # Create tabs for different features
            tabs = st.tabs([
                "AI Chat Assistant", 
                "Component Image Recognition", 
                "Fault-to-Spare Recommendation", 
                "Repair Playbook Generator",
                "Live Video Stream",
                "Salesman Mode"  # New tab for Gemini's audio streaming
            ])
            
            # 2. AI Chat Assistant Tab
            with tabs[0]:
                st.markdown('<div class="feature-header">AI Chat Assistant</div>', unsafe_allow_html=True)
                
                # Quick access buttons for common queries
                st.markdown("#### Quick Access")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("Error Codes"):
                        new_message = "What are the common error codes for this machine and how do I fix them?"
                        st.session_state.chat_history.append({"role": "user", "content": new_message})
                        
                        # Create system prompt
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        
                        # Get response
                        with st.spinner("Generating response..."):
                            if st.session_state.get('stream_chat', True):
                                full_response = stream_response(
                                    [system_prompt, new_message],
                                    st.session_state.flash_model
                                )
                            else:
                                response = st.session_state.model.generate_content([system_prompt, new_message])
                                full_response = response.text
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                with col2:
                    if st.button("Parts List"):
                        new_message = "What are the main parts of this machine and their part numbers?"
                        st.session_state.chat_history.append({"role": "user", "content": new_message})
                        
                        # Create system prompt
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        
                        # Get response
                        with st.spinner("Generating response..."):
                            if st.session_state.get('stream_chat', True):
                                full_response = stream_response(
                                    [system_prompt, new_message],
                                    st.session_state.flash_model
                                )
                            else:
                                response = st.session_state.model.generate_content([system_prompt, new_message])
                                full_response = response.text
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                with col3:
                    if st.button("Safety Precautions"):
                        new_message = "What safety precautions should I follow when using this machine?"
                        st.session_state.chat_history.append({"role": "user", "content": new_message})
                        
                        # Create system prompt
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        
                        # Get response
                        with st.spinner("Generating response..."):
                            if st.session_state.get('stream_chat', True):
                                full_response = stream_response(
                                    [system_prompt, new_message],
                                    st.session_state.flash_model
                                )
                            else:
                                response = st.session_state.model.generate_content([system_prompt, new_message])
                                full_response = response.text
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                with col4:
                    if st.button("Maintenance Schedule"):
                        new_message = "What is the recommended maintenance schedule for this machine?"
                        st.session_state.chat_history.append({"role": "user", "content": new_message})
                        
                        # Create system prompt
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        
                        # Get response
                        with st.spinner("Generating response..."):
                            if st.session_state.get('stream_chat', True):
                                full_response = stream_response(
                                    [system_prompt, new_message],
                                    st.session_state.flash_model
                                )
                            else:
                                response = st.session_state.model.generate_content([system_prompt, new_message])
                                full_response = response.text
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # Chat interface
                st.markdown("#### Chat with the AI Assistant")
                
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                
                # Initialize chat_input value in session state if it doesn't exist
                if "chat_input" not in st.session_state:
                    st.session_state.chat_input = ""
                
                # Chat input - use a callback to handle submission
                def submit_chat():
                    if st.session_state.chat_input:
                        user_message = st.session_state.chat_input
                        st.session_state.chat_history.append({"role": "user", "content": user_message})
                        
                        # Create system prompt
                        system_prompt = create_system_prompt(selected_machine, manual_path)
                        
                        # Clear the input
                        clear_chat_input()
                        
                        # Get response
                        with st.spinner("Generating response..."):
                            if st.session_state.get('stream_chat', True):
                                full_response = stream_response(
                                    [system_prompt, user_message],
                                    st.session_state.flash_model
                                )
                            else:
                                response = st.session_state.model.generate_content([system_prompt, user_message])
                                full_response = response.text
                                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        
                        # Force a rerun to update the UI
                        st.rerun()
                
                # Text input with on_change callback
                user_input = st.text_input(
                    "Ask about error codes, parts, troubleshooting, etc.",
                    key="chat_input",
                    on_change=submit_chat
                )
                
                # Stream option
                st.checkbox("Stream response", value=True, key="stream_chat")
                
                # Add a separate send button for users who prefer clicking
                if st.button("Send"):
                    # The submit_chat function will be triggered if there's text in the input
                    if st.session_state.chat_input:
                        submit_chat()
            
            # 3. Component Image Recognition Tab
            with tabs[1]:
                st.markdown('<div class="feature-header">Component Image Recognition</div>', unsafe_allow_html=True)
                st.markdown("""
                Upload a photo of a component or circuit board, and I'll identify it 
                and provide relevant information from the manual.
                """)
                
                uploaded_image = st.file_uploader("Upload component image", type=["jpg", "jpeg", "png"])
                
                if uploaded_image is not None:
                    # Display the uploaded image
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Component", width=400)
                    
                    if st.button("Identify Component"):
                        with st.spinner("Analyzing component..."):
                            analysis_result = analyze_component_image(
                                image, 
                                selected_machine, 
                                manual_path, 
                                st.session_state.model
                            )
                            
                            st.markdown("### Component Analysis")
                            st.markdown(analysis_result)
            
            # 4. Fault-to-Spare Recommendation Tab
            with tabs[2]:
                st.markdown('<div class="feature-header">Fault-to-Spare Recommendation</div>', unsafe_allow_html=True)
                st.markdown("""
                Upload a fault list PDF, and I'll analyze it to recommend the most likely 
                spare parts needed for repair.
                """)
                
                uploaded_fault_list = st.file_uploader("Upload fault list PDF", type=["pdf"])
                
                if uploaded_fault_list is not None:
                    st.success(f"Uploaded: {uploaded_fault_list.name}")
                    
                    if st.button("Analyze Faults"):
                        with st.spinner("Analyzing fault list and recommending spare parts..."):
                            analysis_result = process_fault_list(
                                uploaded_fault_list,
                                selected_machine,
                                manual_path,
                                st.session_state.model
                            )
                            
                            st.markdown("### Fault Analysis and Spare Parts Recommendation")
                            st.markdown(analysis_result)
            
            # 5. Repair Playbook Generator Tab
            with tabs[3]:
                st.markdown('<div class="feature-header">Repair Playbook Generator</div>', unsafe_allow_html=True)
                st.markdown("""
                Describe the issue or select a component, and I'll generate a 
                step-by-step repair guide with safety precautions.
                """)
                
                issue_description = st.text_area(
                    "Describe the issue or component to repair",
                    placeholder="E.g., 'E21 error code on display', 'Replace the main PCB', 'Wire feed motor not working'"
                )
                
                if st.button("Generate Repair Playbook") and issue_description:
                    with st.spinner("Generating repair playbook..."):
                        playbook = generate_repair_playbook(
                            issue_description,
                            selected_machine,
                            manual_path,
                            st.session_state.model
                        )
                        
                        st.markdown("### Repair Playbook")
                        st.markdown(playbook)
            
            # 6. Live Video Streaming Tab
            with tabs[4]:
                st.markdown('<div class="feature-header">Live Video Stream</div>', unsafe_allow_html=True)
                
                st.markdown("""
                Use your camera to stream live video for remote diagnostics or share your screen 
                to show the machine's display. You can also take screenshots for AI analysis.
                """)
                
                # Create two columns for the video options
                stream_col1, stream_col2 = st.columns(2)
                
                # Option 1: Live Webcam Stream
                with stream_col1:
                    st.markdown("#### Camera Stream")
                    
                    # Configure WebRTC
                    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                    video_processor = VideoProcessor()
                    
                    webrtc_ctx = webrtc_streamer(
                        key="video-stream",
                        video_processor_factory=lambda: video_processor,
                        rtc_configuration=rtc_config,
                        media_stream_constraints={"video": True, "audio": False},
                    )
                    
                    if webrtc_ctx.video_processor:
                        if st.button("Take Screenshot"):
                            webrtc_ctx.video_processor.toggle_screenshot()
                            st.info("Screenshot captured! Check below to analyze it.")
                            
                        # Display and analyze screenshot if available
                        screenshot = webrtc_ctx.video_processor.get_screenshot()
                        if screenshot is not None:
                            st.image(screenshot, caption="Captured Screenshot", width=400)
                            
                            if st.button("Analyze Screenshot"):
                                # Convert OpenCV image to PIL for analysis
                                pil_image = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                                
                                with st.spinner("Analyzing screenshot..."):
                                    analysis_result = analyze_component_image(
                                        pil_image,
                                        selected_machine,
                                        manual_path,
                                        st.session_state.model
                                    )
                                    
                                    st.markdown("### Screenshot Analysis")
                                    st.markdown(analysis_result)
                
                # Option 2: Screen Sharing
                with stream_col2:
                    st.markdown("#### Screen Sharing")
                    st.markdown("""
                    For desktop/screen sharing, click the button below to start sharing your screen.
                    This allows you to show error messages or settings on the machine's display.
                    """)
                    
                    # NOTE: Due to Streamlit's limitations, we're implementing this with a placeholder button
                    # that would normally launch a screen sharing session using a JavaScript library
                    if st.button("Share Your Screen"):
                        st.markdown("""
                        Screen sharing initiated.
                        
                        Select which window or screen to share when prompted by your browser.
                        """)
                        
                        # Here you would normally include JavaScript to start screen sharing
                        # In a production application, you would use a WebRTC library for screen sharing
                        # This is just a placeholder for the actual implementation
                        st.info("Note: Screen sharing simulation only. In a production app, this would launch a WebRTC screen sharing session.")
            
            # 7. Salesman Mode Tab (New)
            with tabs[5]:
                st.markdown('<div class="feature-header">Salesman Assistant Mode</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="salesman-mode">
                This mode uses Gemini Live API for real-time audio conversations between potential customers 
                and an AI-powered sales assistant. The AI can speak naturally about the selected ESAB welding 
                machine's features, benefits, and technical specifications.
                </div>
                """, unsafe_allow_html=True)
                
                # Instructions
                st.markdown("""
                ### How Gemini Live Audio Works:
                1. **Real-time WebSocket Connection**: Establishes a persistent connection with Gemini Live API
                2. **Direct Audio Streaming**: Your microphone audio is sent as PCM data to the API
                3. **Instant Voice Response**: Gemini responds with natural speech played through your speakers
                4. **No File Processing**: Audio is streamed in real-time chunks, not saved as files
                
                ### To Use Salesman Mode:
                1. Ensure your microphone and speakers are working
                2. Click 'Start Live Conversation' to begin the AI sales session
                3. Speak naturally - the AI will respond in real-time
                4. The AI is programmed to highlight features and benefits of the selected machine
                """)
                
                # API Key Check
                api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("""
                    **Google API Key Required**: This feature requires a Google API key with access to Gemini Live API.
                    Please add your API key to Streamlit secrets or environment variables.
                    """)
                else:
                    st.success("✅ Google API key detected - Ready for Gemini Live")
                
                # Start/Stop buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🎙️ Start Live Conversation"):
                        if not api_key:
                            st.error("Cannot start - API key missing")
                        else:
                            if not st.session_state.gemini_live_audio:
                                # Initialize the audio processing class
                                st.session_state.gemini_live_audio = GeminiLiveAudio(
                                    selected_machine,
                                    manual_path
                                )
                            
                            # Start the audio processing
                            if st.session_state.gemini_live_audio.start():
                                st.success("🎯 Live conversation started! Start speaking with the AI salesman.")
                                st.info("The AI has been briefed on the " + selected_machine + " and is ready to discuss its features.")
                            else:
                                st.error("Failed to start live conversation. Please check your microphone permissions.")
                
                with col2:
                    if st.button("⏹️ Stop Conversation"):
                        if st.session_state.gemini_live_audio:
                            if st.session_state.gemini_live_audio.stop():
                                st.info("Live conversation stopped.")
                            else:
                                st.error("Failed to stop the conversation.")
                
                # Status indicator
                if st.session_state.gemini_live_audio and st.session_state.gemini_live_audio.running:
                    st.success("🔴 LIVE: AI Salesman is listening and ready to respond")
                else:
                    st.info("⚫ OFFLINE: Start the conversation to begin")
                
                # Technical Details
                with st.expander("🔧 Technical Implementation Details"):
                    st.markdown("""
                    **What's Different About This Implementation:**
                    
                    - **Real Gemini Live API**: Uses `google.genai.Client` with Live API endpoints
                    - **WebSocket Connection**: Persistent connection via `client.aio.live.connect()`
                    - **PCM Audio Streaming**: Raw audio data sent as `{"data": audio_bytes, "mime_type": "audio/pcm"}`
                    - **Async Processing**: Multiple async tasks for audio capture, sending, receiving, and playback
                    - **No PyAudio File I/O**: Direct stream-to-stream audio processing
                    
                    **Audio Configuration:**
                    - Input: 16kHz mono PCM from microphone
                    - Output: 24kHz mono PCM to speakers  
                    - Chunk size: 1024 samples
                    - Model: `gemini-2.0-flash-live-001`
                    - Voice: "Puck" (configurable)
                    """)
                
                # Conversation History
                st.markdown("### 💬 Conversation History")
                
                # Display conversation history
                if st.session_state.gemini_live_audio and st.session_state.gemini_live_audio.conversation_history:
                    conversation_container = st.container()
                    with conversation_container:
                        for exchange in st.session_state.gemini_live_audio.conversation_history[-10:]:  # Show last 10 exchanges
                            if exchange["role"] == "user":
                                st.markdown(f'<div class="chat-message user-message"><strong>Customer:</strong> {exchange["content"]}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="chat-message assistant-message"><strong>AI Salesman:</strong> {exchange["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.info("Conversation history will appear here once you start talking with the AI salesman.")
                
                # Alternative text input
                st.markdown("### ⌨️ Text Input Alternative")
                st.info("While the AI is designed for voice conversation, you can also type questions:")
                
                text_question = st.text_input("Type a question about the " + selected_machine)
                if st.button("Send Text Question") and text_question:
                    if st.session_state.gemini_live_audio and st.session_state.gemini_live_audio.running:
                        response = st.session_state.gemini_live_audio.send_text_input(text_question)
                        st.success(f"✅ {response}")
                        # Force a rerun to show updated conversation
                        st.rerun()
                    else:
                        st.warning("Please start the live conversation first to send text messages.")

# Run the app
if __name__ == "__main__":
    main()
