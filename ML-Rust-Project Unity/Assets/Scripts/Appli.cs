using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

public class Appli : MonoBehaviour
{

    [SerializeField]
    private string requestPicPath;

    [SerializeField] private int epochs;

    [Space(10)]

    [Header("Input Value in float")]

    [SerializeField] private int[] npl = new[] { 4, 4 };
    private int numberLayer;


    [SerializeField]
    private string userRequestPathFile;


    [Range(0.0f, 1.0f)]
    [SerializeField] private double learningRate = 0.8f;

    [SerializeField] private bool isClassification = true;

    private List<double> inputList = new List<double>();

    private List<double> outputList = new List<double>();

    [SerializeField]
    double[] requestInput = new double[32 * 32 * 3];

    double[] trainningInput;

    double[] trainningOuput;

    private System.IntPtr MyModel;

    private int _inputSize;
    private int _outputSize;



    public static Texture2D Resize(Texture2D source, int newWidth, int newHeight)
    {
        source.filterMode = FilterMode.Point;
        RenderTexture rt = RenderTexture.GetTemporary(newWidth, newHeight);
        rt.filterMode = FilterMode.Point;
        RenderTexture.active = rt;
        Graphics.Blit(source, rt);
        Texture2D nTex = new Texture2D(newWidth, newHeight);
        nTex.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        nTex.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return nTex;
    }

    public static void SetTextureImporterFormat(Texture2D texture, bool isReadable)
    {
        if (null == texture) return;

        string assetPath = AssetDatabase.GetAssetPath(texture);
        var tImporter = AssetImporter.GetAtPath(assetPath) as TextureImporter;
        if (tImporter != null)
        {
            tImporter.textureType = TextureImporterType.Default;

            tImporter.isReadable = isReadable;

            AssetDatabase.ImportAsset(assetPath);
            AssetDatabase.Refresh();
        }
    }

    void requestPicRead()
    {
        Texture2D requestImage = Resources.Load<Texture2D>(requestPicPath);

        SetTextureImporterFormat(requestImage, true);

        Texture2D image = Resize(requestImage, 32, 32);

        SetTextureImporterFormat(image, true);





        int count = 0;

        if (image)
        {

            SetTextureImporterFormat(image, true);

            for (int i = 0; i < image.width; i++)
            {
                for (int j = 0; j < image.height; j++)
                {
                    Color pixel = image.GetPixel(i, j);

                    requestInput[count] = pixel.r;

                    count = count + 1;

                    requestInput[count] = pixel.g;

                    count = count + 1;

                    requestInput[count] = pixel.b;

                    count = count + 1;

                }
            }

        }

       else
        {
            Debug.Log("bite");

        }

    }
 

    void predictMLPMulticlass()
    {

            int bestModel = 0;
            double bestValue = 0.0f;

            IntPtr pointerToValues = MlDllWrapper.PredictMLPModel(MyModel, requestInput, _inputSize,
                numberLayer, npl, isClassification);

            string[] arrayValuesasString = Marshal.PtrToStringAnsi(pointerToValues).Split(';');

            for (int i = 0; i < _outputSize; i++)
            {
                double currentValue = Convert.ToDouble(arrayValuesasString[i].Replace('.', ','));


                print($"{currentValue} and model {i}");
                //print($"{currentValue} and model {i}");
                if (currentValue > bestValue)
                {
                    bestModel = i;
                    bestValue = currentValue;
                }
            }

            print($"The best value is {bestValue} with the model {bestModel}");

        //Regression Lineaire

        /*if(pointerToValues != IntPtr.Zero)
            MlDllWrapper.DeleteLinearModel(pointerToValues);*/
    }

    // Start is called before the first frame update
    void Start()
    {

        CsvReader.readCSVFile("TrainningData\\inputCsv.csv", ref inputList, ref outputList);

        trainningInput = inputList.ToArray();
        trainningOuput = outputList.ToArray();

        _inputSize = npl[0];
        _outputSize = npl.Last();
        numberLayer = npl.Length;

        if (CsvReader.inputCount != _inputSize)
            Debug.LogError($"Input Length ({CsvReader.inputCount}) in CSV File don't match the npl input length ({_inputSize})");

        if (CsvReader.outputCount != _outputSize)
            Debug.LogError($"Output Length ({CsvReader.outputCount}) in CSV File don't match the npl output length ({_outputSize})");

        MyModel = MlDllWrapper.CreateMLPModel(numberLayer, npl);

    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            predictMLPMulticlass();
        }

        if (Input.GetKeyDown(KeyCode.I))
        {
            File.Copy(userRequestPathFile, @".\Assets\Resources\Dataset\requestImage.jpg", true);
            requestPicRead();
        }

        if (Input.GetKeyDown(KeyCode.T))
        {
            print("Starting trainning");
            MlDllWrapper.trainMLPModelClass(MyModel, numberLayer, trainningInput.Length / npl[0], npl, trainningInput,
                trainningInput.Length, _inputSize, trainningOuput, trainningOuput.Length, _outputSize, epochs,
                learningRate, isClassification);
            print("Trainning Finished");
            
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            MlDllWrapper.DeleteLinearModel(MyModel);
        }
    }
}
