using UnityEngine;
using System.Collections.Generic;
using System.Diagnostics;



public class ExecutePathfinding : MonoBehaviour
{

    //UnityEngine.Debug.Log($"Error");
    public float speed = 1f; // Movement speed
    private int currentPointIndex = 0; // Index of the current point in the path
    private List<Vector3> coordinates;

    void Start()
    {
        string roboSimPath = Application.dataPath + "/RoboSim";
        string sceneDataPath = roboSimPath+"/bin/sceneData.json";
        string waypointPath = roboSimPath+"/bin/waypoints.txt";

        ComputePath computePath = new ComputePath();
        computePath.SaveSceneLayout(sceneDataPath);

        Stopwatch timePathfinding = new Stopwatch();
        timePathfinding.Start();
        coordinates = computePath.ComputePathFinding(sceneDataPath, waypointPath);
        timePathfinding.Stop();

        //mark the path in unity with small red squares
        foreach (Vector3 coordinate in coordinates)
        {
            Quaternion spawnRotation = Quaternion.identity; // No rotation, use default
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f); //make the cube smaller
            cube.GetComponent<Renderer>().material.color = Color.red; //change the color to red
            Instantiate(cube, coordinate, spawnRotation);
        }

        double totalDistance = getTotalDistance(coordinates);
        print($"Distance Traveled {totalDistance}ft\nTime to compute path {timePathfinding.ElapsedMilliseconds}ms");
    }

    void Update()
    {
        // Check if we reached the end of the path
        if (currentPointIndex < coordinates.Count)
        {
            // Calculate the direction to the next point
            Vector3 targetPosition = coordinates[currentPointIndex];
            Vector3 direction = (targetPosition - transform.position).normalized;

            // Move towards the next point based on speed
            transform.position += direction * speed * Time.deltaTime;

            // Check if we are close enough to the current point to move to the next
            if (Vector3.Distance(transform.position, targetPosition) < 0.1f)
            {
                currentPointIndex++;
            }
        }
    }

    double getTotalDistance(List<Vector3> coordinates)
    {
        double totalDistance = 0;
        for (int i = 0; i < coordinates.Count-1; i++)
        {
            Vector3 coordinate1 = coordinates[i];
            Vector3 coordinate2 = coordinates[i+1];
            totalDistance += Vector3.Distance(coordinate1, coordinate2);
        }
        return totalDistance;
    }
}
