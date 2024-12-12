using UnityEngine;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using System;

public class ComputePath : MonoBehaviour
{
    public void SaveSceneLayout(string filePath)
    {
        GameObject[] allObjects = FindObjectsOfType<GameObject>();

        var SceneColliders = new Dictionary<string, Dictionary<string, Vector3>>();

        foreach (GameObject obj in allObjects)
        {
            Collider collider = obj.GetComponent<Collider>();
            Vector3 dimensions = Vector3.zero;
            if (collider != null) {
                dimensions = collider.bounds.size;
            }
            if (collider != null && dimensions != Vector3.zero)
            {
                var colliderData = new Dictionary<string, Vector3>
                {
                    {"position", collider.bounds.center},
                    {"dimension", collider.bounds.size}
                };
                SceneColliders[obj.name] = colliderData;
            }
        }

        // Convert the dictionary to JSON
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new Vector3JsonConverter() }
        };
        string json = JsonSerializer.Serialize(SceneColliders, options);

        // Output the JSON
        Console.WriteLine(json);
        File.WriteAllText(filePath, json);

        UnityEngine.Debug.Log("Scene data exported to: " + filePath);
    }

    public List<Vector3> ComputePathFinding(string sceneDataPath, string waypointPath)
    {
        string algorithmPath = "algorithm.py";
        string dockerContainerTag = "pathfinding";

        //parse the provided paths to aquire their relative paths (from the perspective of the RoboSim dir)
        //This is necessary as they are provided to the container's filesystem
        string relativeSceneDataPath = sceneDataPath.Substring(sceneDataPath.IndexOf("bin"));
        string relativeWaypointPath = waypointPath.Substring(waypointPath.IndexOf("bin"));

        string pwd = Application.dataPath+"/RoboSim";
        string dockerRun = "run -i"; //run and track output
        string dockerAddVolume =  "-v "+pwd+":/src:rw"; //copy Robosim's contents to the container
        string dockerContainerCommand = "python "+algorithmPath+" "+relativeSceneDataPath+" "+relativeWaypointPath; //what will run on the container
        string dockerCommand = string.Join(" ", new string[] {dockerRun, dockerAddVolume, dockerContainerTag, dockerContainerCommand});

        UnityEngine.Debug.Log($"cmd {dockerCommand}");

        //Create a new process to execute the Docker command
        Process process = new Process();
        process.StartInfo.FileName = "docker";
        process.StartInfo.Arguments = $"{dockerCommand}";
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true; 

        //output and error logging from the container
        process.OutputDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                UnityEngine.Debug.Log($"Output: {e.Data}");
            }
        };
        process.ErrorDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                UnityEngine.Debug.Log($"Error: {e.Data}");
            }
        };

        //run the process
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        process.WaitForExit();

        //Ensure the container succeeded
        if (process.ExitCode == 0)
        {
            UnityEngine.Debug.Log("Pathfinding container ran successfully!");
        }
        else
        {
            UnityEngine.Debug.Log($"Docker process exited with code {process.ExitCode}");
        }

        return LoadCoordinates(waypointPath);
    }

    private List<Vector3> LoadCoordinates(string waypointPath)
    {
        List<Vector3> points = new List<Vector3>();
        string[] lines = File.ReadAllLines(waypointPath);

        for (int i = 0; i < lines.Length; i++)
        {
            string[] values = lines[i].Split(',');
            if (values.Length == 3 &&
                float.TryParse(values[0], out float x) &&
                float.TryParse(values[1], out float y) &&
                float.TryParse(values[2], out float z))
            {
                points.Add(new Vector3(x, y, z));
            }
        }
        return points;
    }
}


// Custom JSON converter for Vector3
public class Vector3JsonConverter : JsonConverter<Vector3>
{
    public override Vector3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
            throw new JsonException();

        float x = 0, y = 0, z = 0;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
                return new Vector3(x, y, z);

            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                string propertyName = reader.GetString();
                reader.Read();

                switch (propertyName)
                {
                    case "X":
                        x = reader.GetSingle();
                        break;
                    case "Y":
                        y = reader.GetSingle();
                        break;
                    case "Z":
                        z = reader.GetSingle();
                        break;
                }
            }
        }

        throw new JsonException();
    }

    public override void Write(Utf8JsonWriter writer, Vector3 value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("X", value.x);
        writer.WriteNumber("Y", value.y);
        writer.WriteNumber("Z", value.z);
        writer.WriteEndObject();
    }
}