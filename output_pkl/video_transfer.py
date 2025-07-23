import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
import os
import tempfile


# Function to extract audio from a video file
def extract_audio_from_video(video_file_path, output_format='mp3'):
    try:
        video_clip = VideoFileClip(video_file_path)
        if video_clip.audio is None:
            st.error("The video file does not contain an audio track.")
            return None
        audio_file = f"extracted_audio.{output_format}"
        video_clip.audio.write_audiofile(audio_file)
        video_clip.close()  # Close the video file to release it
        return audio_file
    except Exception as e:
        st.error(f"An error occurred while extracting audio from the video: {e}")
        return None


# Function to convert audio to text
def convert_audio_to_text(audio_file_path, language="mr-IN"):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Use tempfile for unique WAV file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_file_path = tmp_file.name
        audio.export(wav_file_path, format="wav")

    # Open the WAV file
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)

    # Recognize speech using Google's API
    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    finally:
        # Clean up the temporary WAV file
        if os.path.exists(wav_file_path):
            try:
                os.remove(wav_file_path)
            except PermissionError:
                pass  # If file is still locked, pass for now


def main():
    st.title("Media Processing App")
    st.sidebar.title("Choose a Service")
    service_option = st.sidebar.selectbox(
        "Select an option",
        ("Separate Audio from Video", "Extract Text from Audio", "Do all Automatically")
    )

    if service_option == "Separate Audio from Video":
        st.header("Separate Audio from Video")
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        output_format = st.selectbox("Select output audio format", ("mp3", "wav", "ogg"))

        if uploaded_video is not None:
            with st.spinner("Extracting audio..."):
                # Save the uploaded video to a temporary file
                video_file_path = "temp_video.mp4"
                with open(video_file_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                # Extract audio from the video
                extracted_audio_file = extract_audio_from_video(video_file_path, output_format)
                if extracted_audio_file:
                    st.success("Audio extracted successfully!")
                    st.audio(extracted_audio_file)

                # Clean up the temporary video file
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)

    elif service_option == "Extract Text from Audio":
        st.header("Extract Text from Audio")
        uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

        if uploaded_audio is not None:
            with st.spinner("Transcribing audio..."):
                # Save the uploaded audio to a temporary file
                audio_file_path = "temp_audio.mp3"
                with open(audio_file_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())

                transcribed_text = convert_audio_to_text(audio_file_path, language="mr-IN")
                st.success("Transcription completed successfully!")
                st.text_area("Transcribed Text", transcribed_text)

                # Clean up the temporary audio file
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)

    elif service_option == "Do all Automatically":
        st.header("Do all Automatically")
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            with st.spinner("Processing video..."):
                # Save the uploaded video to a temporary file
                video_file_path = "temp_video.mp4"
                with open(video_file_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                # Step 1: Extract audio from the video
                extracted_audio_file = extract_audio_from_video(video_file_path, output_format='mp3')
                if extracted_audio_file:
                    # Step 2: Convert extracted audio to text
                    transcribed_text = convert_audio_to_text(extracted_audio_file, language="mr-IN")

                    # Display the transcribed text
                    st.success("Audio extraction and transcription completed successfully!")
                    st.text_area("Transcribed Text", transcribed_text)

                    # Step 3: Save the transcribed text to a file and provide download option
                    text_file_path = "transcribed_text.txt"
                    with open(text_file_path, "w", encoding="utf-8") as text_file:
                        text_file.write(transcribed_text)

                    # Download link for the text file
                    with open(text_file_path, "rb") as file:
                        st.download_button(
                            label="Download Transcribed Text",
                            data=file,
                            file_name="transcribed_text.txt",
                            mime="text/plain"
                        )

                    # Clean up the temporary files
                    if os.path.exists(text_file_path):
                        os.remove(text_file_path)

                # Clean up the temporary video and audio files
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)
                if extracted_audio_file and os.path.exists(extracted_audio_file):
                    os.remove(extracted_audio_file)


if __name__ == "__main__":
    main()
