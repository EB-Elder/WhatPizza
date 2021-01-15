using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Collections;
using UnityEditor;
using System.Text;

struct Pizzatype
{
    public string path;
    public string output;

    public Pizzatype(string x, string y)
    {
        this.path = x;
        this.output = y;
    }
}


public class PicReader : MonoBehaviour
{

    [SerializeField]
    List<Pizzatype> directoryList; 


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

    // Start is called before the first frame update
    void Start()
    {

        string strFilePath = @"Assets\Resources\Dataset\inputCsv.csv";
        string strSeperator = ";";
        StringBuilder csvBuilder = new StringBuilder();
        string line = "";
        string entete = "";

        directoryList = new List<Pizzatype>() {
        new Pizzatype(@"Resources\Dataset\pizza calzone", "0;0;0;0;0;1"),
        new Pizzatype(@"Resources\Dataset\pizza chèvre miel","0;0;0;0;1;0"),
        new Pizzatype(@"Resources\Dataset\pizza hawaiana", "0;0;0;1;0;0"),
        new Pizzatype(@"Resources\Dataset\pizza margherita", "0;0;1;0;0;0"),
        new Pizzatype(@"Resources\Dataset\pizza regina","0;1;0;0;0;0"),
        new Pizzatype(@"Resources\Dataset\pizza4fromages", "1;0;0;0;0;0")
        };

        for (int i = 0; i < 32*32*3; i = i+1)
        {
            entete = entete + "X" + i + ";";
        }

        entete = entete + "Y0;Y1;Y2;Y3;Y4;Y5";

        csvBuilder.AppendLine(entete);


        foreach (Pizzatype link in directoryList)
        {

            string[] fileEntries = Directory.GetFiles(@"Assets\" + link.path,"*.jpg");


            

            foreach (string linkimage in fileEntries)
            {

                Texture2D image = Resources.Load<Texture2D>(linkimage.Substring(17, linkimage.Length - 21));
                line = "";
             
 
                if(image)
                {

                    SetTextureImporterFormat(image, true);

                    for (int i = 0; i < image.width; i++)
                    {
                        for (int j = 0; j < image.height; j++)
                        {
                            Color pixel = image.GetPixel(i, j);

                            line = line + pixel.r + ";" + pixel.g + ";" + pixel.b + ";";



                        }
                    }

                    line = line + link.output;
                    csvBuilder.AppendLine(line);
                }

                else
                {
                    Debug.Log("bite");
                }



            }



            

        }

        File.WriteAllText(strFilePath, csvBuilder.ToString());







    }


}
