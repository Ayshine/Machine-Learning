<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Default.aspx.cs" Inherits="_Default" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        <table id="Table1" runat="server">
            <tr>
                <td>
                    <asp:Label ID="Select" runat="server" Text="Select your Image to be classified"></asp:Label>

                </td>
                <td>
                  
                </td>

            </tr>
             <tr>
                <td>
                     <asp:FileUpload ID="UploadImages" runat="server" />
                    <asp:Button ID="Button1" runat="server" Text="Upload" OnClick="UploadBTN_Click" />
                </td>
                <td>                   
                </td>

            </tr>
             <tr>
                <td>
                    <asp:Literal ID="Classify" runat="server"> Classify your image</asp:Literal>
                </td>
                <td>
                    <asp:Button ID="Button2" runat="server" Text="Classify" OnClick="Button2_Click" />
                </td>

            </tr>
            <tr>
                <td>
                    <asp:Literal ID="CNNResultLT" runat="server">CNN Results</asp:Literal>
                </td>
                <td>
                    <asp:Literal ID="TransferResultLT" runat="server">Transfer Learning Results</asp:Literal>
                </td>

            </tr>
            <tr>
                <td>
                    <asp:TextBox ID="CNNResultTXT" runat="server"></asp:TextBox>
                </td>
                <td>
                    <asp:TextBox ID="TranferResultTXT" runat="server"></asp:TextBox>
                </td>

            </tr>
        </table>

    </form>
</body>
</html>
