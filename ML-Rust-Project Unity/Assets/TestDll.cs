using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

public class TestDll : MonoBehaviour
{
    [SerializeField]
    private sphereExposer[] spheres;

    [SerializeField] private Material blueMat;
    [SerializeField] private Material redMat;
    [SerializeField] private Material greenMat;

    [SerializeField] private int epochs;
    
    [Space(10)]
    
    [Header("Input Value in float")]
    
    [SerializeField] private int[] npl = new[] {4, 4};
    [SerializeField] private double[] inputs = new[] {1.0, 1.0}; 
    private int numberLayer;
    
    
    [Range(0.0f, 1.0f)] 
    [SerializeField] private double learningRate = 0.8f;

    [SerializeField] private bool isClassification = true;
    
    private List<double> inputList = new List<double>();

    private List<double> outputList = new List<double>();
    
    double[] trainningInput;

    double[] trainningOuput;

    // Start is called before the first frame update

    private int modelSize = 2;
    private System.IntPtr MyModel;
    List<double[]> testDataSet = new List<double[]>();

    private List<System.IntPtr> _myMulticlassModel = new List<IntPtr>();

    private List<double> _multiclassResult = new List<double>();
    List<double[]> _outputTrainningMulticlass = new List<double[]>();
    
    private int _inputSize;
    private int _outputSize;
    
    void randomizeSpheres()
    {
        foreach (sphereExposer sphere in spheres)
        {
            sphere.myTransform.position = new Vector3(Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), 0.0f);
        }
    }

    void CreateLinearMulticlass()
    {
        for (int i = 0; i < _outputSize; i++)
        {
            _myMulticlassModel.Add(MlDllWrapper.CreateLinearModel(_inputSize));
        }
    }

    void predictLinearMulticlass()
    {
        
        
            
        
        for (int j = 0; j < testDataSet.Count; j++)
        {
            
            _multiclassResult.Clear();

            int bestModel = 0;
            double bestValue = 0.0f;
            for (int i = 0; i < _outputSize; i++)
            {
                double currentValue = MlDllWrapper.PredictLinearModel(_myMulticlassModel[i], testDataSet[j], _inputSize, isClassification);

                print($"Model numero {i} = {currentValue}");
                if (currentValue > bestValue)
                {
                    bestModel = i;
                    bestValue = currentValue;
                }
                
            }

            switch (bestModel)
            {
                default:
                    spheres[j].ChangeMaterial(blueMat);
                    break;
                case 1:
                    spheres[j].ChangeMaterial(redMat);
                    break;
                case 2:
                    spheres[j].ChangeMaterial(greenMat);
                    break;
            }

        }
        
        
        
    }

    void trainLinearMulticlass()
    {
        
        
        
        print("Starting trainning");
        _outputTrainningMulticlass.Clear();
        
        _outputTrainningMulticlass = new List<double[]>();

        int outputLength = trainningOuput.Length / _outputSize;
        
        for (int i = 0; i < _outputSize; i++)
        {
            List<double> tmp = new List<double>();
            for (int j = 0; j < outputLength; j++)
            {
                tmp.Add(trainningOuput[j * _outputSize + i]);
            }
            _outputTrainningMulticlass.Add(tmp.ToArray());
        }
        

        for (int i = 0; i < _outputTrainningMulticlass.Count; i++)
        {
            MlDllWrapper.trainLinearClass(_myMulticlassModel[i], trainningInput, trainningInput.Length, _inputSize, _outputTrainningMulticlass[i],
                    _outputTrainningMulticlass[i].Length, 1, learningRate, isClassification, epochs);
        }
        
        print("Trainning Finished");
    }

    void predictMLPMulticlass()
    {

        for (int j = 0; j < testDataSet.Count; j++)
        {
            
            _multiclassResult.Clear();

            int bestModel = 0;
            double bestValue = 0.0f;
            
            IntPtr pointerToValues = MlDllWrapper.PredictMLPModel(MyModel, testDataSet[j], _inputSize,
                numberLayer, npl, isClassification); 
            
            string [] arrayValuesasString = Marshal.PtrToStringAnsi(pointerToValues).Split(';');

            for (int i = 0; i < _outputSize; i++)
            {
                double currentValue = Convert.ToDouble(arrayValuesasString[i].Replace('.', ','));
                
                
                print($"{currentValue} and model {i}");
                if (currentValue > bestValue)
                {
                    bestModel = i;
                    bestValue = currentValue;
                }
            }
            
            print(arrayValuesasString[0]);
            
            //Regression Lineaire
            spheres[j].changeZ((float) Convert.ToDouble(arrayValuesasString[0].Replace('.', ',')));
            
            switch (bestModel)
            {
                default:
                    spheres[j].ChangeMaterial(blueMat);
                    break;
                case 1:
                    spheres[j].ChangeMaterial(redMat);
                    break;
                case 2:
                    spheres[j].ChangeMaterial(greenMat);
                    break;
            }

            /*if(pointerToValues != IntPtr.Zero)
                MlDllWrapper.DeleteLinearModel(pointerToValues);*/

        }
        
    }

    void predictOnDataSet()
    {
        testDataSet = new List<double[]>();
        
        foreach (sphereExposer sphere in spheres)
        {
            Vector3 pos = sphere.myTransform.position;
            double[] tmp = new[] {(double)pos.x,  pos.y};
            testDataSet.Add(tmp);
        }        
        
        for (int i = 0; i < testDataSet.Count; i++)
        {
            var result = MlDllWrapper.PredictLinearModel(MyModel, testDataSet[i], testDataSet[i].Length, true);
            print("X : " + testDataSet[i][0] + "Y : " + testDataSet[i][1] + " : " + result);
            if (result > 0)
                spheres[i].ChangeMaterial(redMat);
            else
                spheres[i].ChangeMaterial(blueMat);
                
        }
    }

    void trainModel()
    {
        print("Model start trainning");

        MlDllWrapper.trainLinearClass(MyModel, trainningInput, trainningInput.Length, 2,
                trainningOuput, trainningOuput.Length, 1, learningRate, true, epochs);
        print("Model finished train");
        
    }
    
    
    void Start()
    {

        
        
        CsvReader.readCSVFile("TrainningData\\rbf.csv", ref inputList, ref outputList);

        trainningInput = inputList.ToArray();
        trainningOuput = outputList.ToArray();
        


        //MlDllWrapper.InitRBF(trainningInput, trainningInput.Length, 784, trainningOuput, trainningOuput.Length, 10, 500);
        // _inputSize = npl[0];
        // _outputSize = npl.Last();
        // numberLayer = npl.Length;
        //
        // if(CsvReader.inputCount != _inputSize)
        //     Debug.LogError($"Input Length ({CsvReader.inputCount}) in CSV File don't match the npl input length ({_inputSize})");
        //
        // if(CsvReader.outputCount != _outputSize)
        //     Debug.LogError($"Output Length ({CsvReader.outputCount}) in CSV File don't match the npl output length ({_outputSize})");
        // //MyModel = MlDllWrapper.CreateLinearModel(modelSize);
        // //CreateLinearMulticlass();
        //
        //
        // randomizeSpheres();
        //
        // testDataSet = new List<double[]>();
        //
        // foreach (sphereExposer sphere in spheres)
        // {
        //     Vector3 pos = sphere.myTransform.position;
        //     double[] tmp = new[] {(double)pos.x,  pos.y};
        //     testDataSet.Add(tmp);
        // }  
        //
        // //trainModel();
        //
        // //predictOnDataSet();
        //
        // MyModel = MlDllWrapper.CreateMLPModel(numberLayer, npl);

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            /*List<double[]> test = new List<double[]>();

            test.Add(new []{0.0, 1.0});
            test.Add(new []{1.0, 0.0});
            test.Add(new []{0.0, 0.0});
            test.Add(new []{1.0, 1.0});

            for (int i = 0; i < 4; i++)
            {
                print(MlDllWrapper.PredictMLPModel(MyModel, test[i], inputSize, numberLayer, npl, true));
            }*/
            
            
            /*double result = MlDllWrapper.PredictMLPModelClassification(MyModel, inputs, inputSize, numberLayer, npl);
            if (result > 0)
            {
                print(1);
            }
            else
            {
                print(-1);
            }*/
            //predictOnDataSet();
            //predictLinearMulticlass();
            //predictMLPMulticlass();
        }
        
        if (Input.GetKeyDown(KeyCode.T))
        {
            // print("Starting trainning");
            // MlDllWrapper.trainMLPModelClass(MyModel, numberLayer, trainningInput.Length / npl[0], npl, trainningInput,
            //     trainningInput.Length, _inputSize, trainningOuput, trainningOuput.Length, _outputSize, epochs,
            //     learningRate, isClassification);
            // print("Trainning Finished");
            //trainModel();
            //trainLinearMulticlass();
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            MlDllWrapper.DeleteLinearModel(MyModel);
        }
    }
}
