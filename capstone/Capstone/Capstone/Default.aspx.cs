using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class _Default : System.Web.UI.Page
{
   string RecentImagePath = null;
    protected void Page_Load(object sender, EventArgs e)
    {

    }

    protected void UploadBTN_Click(object sender, EventArgs e)
    {
        RecentImagePath = "Uploaded\\" + UploadImages.FileName + ".jpg";
        UploadImages.SaveAs(Server.MapPath(RecentImagePath));

    }


    protected void Button2_Click(object sender, EventArgs e)
    {
        CNNResultTXT.Text = RunPython("../Classifiers/CNN.py");
        TranferResultTXT.Text = RunPython("../Classifiers/Transfer.py");
    }

    public string RunPython(string pythonFileName)
    {
        string PythonApp = pythonFileName + ".py";

        Process process = new Process();
        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;

        // make sure we can read the output from stdout 
        startInfo.UseShellExecute = false;
        startInfo.RedirectStandardOutput = true;

        startInfo.FileName = "Python.exe";
        startInfo.Arguments = "from"+ PythonApp + "import *; hello(" + RecentImagePath + ")";
        process.StartInfo = startInfo;
        process.Start();

        // Read the standard output of the app we called.  
        // in order to avoid deadlock we will read output first 
        // and then wait for process terminate: 
        StreamReader myStreamReader = process.StandardOutput;
        string results = myStreamReader.ReadLine();

        /*if you need to read multiple lines, you might use: 
            string myString = myStreamReader.ReadToEnd() */

        // wait exit signal from the app we called and then close it. 
        process.WaitForExit();
        process.Close();

        // write the output we got from python app 
        return results;

    }
}