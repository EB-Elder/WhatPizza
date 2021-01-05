using System.Runtime.InteropServices;

public static class MlDllWrapper
{
    
    //////////////////////////////////////////////////LINEAR MODEL///////////////////////////////////////////////////////////
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "create_linear_model")]
    public static extern System.IntPtr CreateLinearModel(int inputSize);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_linear_model")]
    public static extern double PredictLinearModel(System.IntPtr model,
        double[] inputs, int inputSize, bool isClassification);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "train_linear_model_class")]
    public static extern void trainLinearClass(System.IntPtr model, double[] inputs, int inputSize, int inputSampleSize,
        double[] ouputs, int outputSize, int outputSampleSize, double learningRate, bool isClassification, int epochs);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_linear_model_multiclass")]
    public static extern System.IntPtr PredictLinearModelMultiClass(System.IntPtr model,
        double[] inputs, int inputSize, int classCount, bool isClassification);

    //////////////////////////////////////////////////MLP MODEL///////////////////////////////////////////////////////////
    
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "create_mlp_model")]
    public static extern System.IntPtr CreateMLPModel(int numberHiddenLayer, int[] neuronesCount);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_mlp_model")]
    public static extern double PredictMLPModel(System.IntPtr model,
        double[] inputs, int inputSize, int numberLayer, int[] npl, bool isClassification);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "train_mlp_model_class")]
    public static extern void trainMLPModelClass(System.IntPtr model, int numberLayer, int datasetSize, int[] npl, double[] input, int inputSize, int inputSampleSize,
        double[] ouputs, int outputSize, int outputSampleSize, int epochs, double learningRate, bool isClassification);
    
    
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "delete_linear_model")]
    public static extern void DeleteLinearModel(System.IntPtr model);
}