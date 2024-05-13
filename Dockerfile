# Base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /pipeline

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /pipeline/src
COPY config /pipeline/config
COPY pipeline.py /pipeline
COPY tests /pipeline/tests

# Run pipeline.py when the container launches
CMD ["python", "pipeline.py"]