FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN sudo apt install tesseract-ocr
RUN export TESSDATA_PREFIX=/usr/share/tesseract-ocr/  

# Copy the rest of the application code
COPY . .

# Command to run the Pathway script
CMD [ "python", "./server.py" ]