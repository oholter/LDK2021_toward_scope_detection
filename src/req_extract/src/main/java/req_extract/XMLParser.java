package req_extract;

import java.io.ByteArrayInputStream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XMLParser {

	public static Document loadXMLFromString(String xml) throws Exception {
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		DocumentBuilder builder = factory.newDocumentBuilder();
		StringBuilder xmlStringBuilder = new StringBuilder();
		xmlStringBuilder.append(xml);
		ByteArrayInputStream input = new ByteArrayInputStream(xmlStringBuilder.toString().getBytes("UTF-8"));
		Document doc = builder.parse(input);
		return doc;
	}

	public static void getRequirements(Document doc) {
		Element e = doc.getDocumentElement();
		System.out.println(e.getNodeName());
		NodeList reqList = doc.getElementsByTagName("req");

		for (int temp = 0; temp < reqList.getLength(); temp++) {
			Node nNode = reqList.item(temp);
			if (nNode.getNodeType() == Node.ELEMENT_NODE) {
				Element eElement = (Element) nNode;
				System.out.println("Req num : " + eElement.getAttribute("num"));
				System.out.println(eElement.getTextContent());
			}
		}
	}

	public static void main(String[] args) throws Exception {
//		String pdfPath = "/home/ole/Documents/dnvgl/ru-ship/dnvgl-st-f101.pdf";
		String pdfPath = "/home/ole/Documents/dnvgl/ru-ship/DNVGL-RU-SHIP-Pt4Ch7.pdf";
		int lastPage = 100;
		PDFParser parser = new PDFParser(pdfPath);
		String xmlString = parser.parseDocument(lastPage);
		Document xmlDocument = loadXMLFromString(xmlString);
		getRequirements(xmlDocument);
	}
}
