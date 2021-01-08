using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using JetBrains.Annotations;
using UnityEngine;

public static class CsvReader
{

    //private static List<double> inputList = new List<double>();

    //private static List<double> outputList = new List<double>();

    //private static String path;

    static List<String> entete = new List<String>();
    
    public static int inputCount = 0;
    public static int outputCount = 0;


    public static void readCSVFile(String path, ref List<double> inputList, ref List<double> outputList)
    {


            string line;

            StreamReader theReader = new StreamReader(path, Encoding.Default);
 
            using (theReader)
            {

                bool drap = true;

                do
                {
                    line = theReader.ReadLine();

                    if (line != null)
                    {
                        string[] entries = line.Split(';');


                        
                        if (drap)
                        {
                            if (entries.Length > 0)
                                for (int i = 0; i < entries.Length; i = i+1)
                                {
                                    entete.Add(entries[i]);
                                    if (entete[i].Substring(0, 1) == "X")
                                        inputCount++;
                                    if (entete[i].Substring(0, 1) == "Y")
                                        outputCount++;
                                }

                            drap = false;
                           
                        }

                        else
                        {

                            for (int i = 0; i < entries.Length; i = i + 1)
                            {

                                if (entete[i].Substring(0, 1) == "X")
                                {

                                    inputList.Add(Convert.ToDouble(entries[i].Replace(".", ",")));

                                }

                                else
                                {

                                    outputList.Add(Convert.ToDouble(entries[i].Replace(".", ",")));

                                }

                            }


                           
                        }

                    }
                }
                while (line != null);

                theReader.Close();

        }
        


    }



}

