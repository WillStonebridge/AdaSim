# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /src

COPY requirements.txt requirements.txt
COPY algorithm.py .
COPY mplanner ./mplanner

#install python libraries
RUN pip install -r requirements.txt



# Set the command to run the script
CMD ["python", "algorithm.py"]
# CMD ["input.txt", "output.txt"]  
# Default args, but can be overridden