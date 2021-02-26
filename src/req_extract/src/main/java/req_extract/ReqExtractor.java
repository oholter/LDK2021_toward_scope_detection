package req_extract;

import java.awt.geom.Rectangle2D;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.PDFTextStripperByArea;
import org.apache.pdfbox.tools.PDFText2HTML;

/**
 * Hello world!
 *
 */
public class ReqExtractor {
	// static String path = "/home/ole/Documents/doc.pdf";
	// static String path = "/home/ole/Documents/report.pdf";

	private String reqPath;
	private File reqFile;
	private PDDocument reqDoc;
	private List<String> reqs;

	private String shallString = "shall";
	private String[] abbreviations = new String[] { "Sec.", "i.e.", "e.g.", "etc.", "Techn.", "Ch.", "Pt.", "1.", "2.",
			"3.", "4.", "5.", "6.", "7.", "8.", "9.", "A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.",
			"L.", "M.", "N.", "O.", "P.", "Q.", "R.", "S.", "T.", "U.", "V.", "W.", "X.", "Y.", "Z." };

	public ReqExtractor(String path) throws Exception {
		this.reqPath = path;
		this.reqFile = new File(reqPath);
		this.reqDoc = PDDocument.load(reqFile);
	}

	public PDDocument getDoc() {
		return reqDoc;
	}

	public List<String> getReqs() {
		return reqs;
	}

	public void close() throws Exception {
		reqDoc.close();
	}

	public void addText(PDDocument doc, PDPage page, String s) throws Exception {
		PDPageContentStream contentStream = new PDPageContentStream(doc, page);

		contentStream.beginText();
		contentStream.setFont(PDType1Font.TIMES_ROMAN, 12);
		contentStream.newLineAtOffset(20, 700);

		contentStream.showText(s);

		contentStream.endText();
		contentStream.close();
	}

	public String readText(PDDocument doc) throws Exception {
		PDFTextStripper stripper = new PDFTextStripper();
		stripper.setStartPage(18);
		stripper.setEndPage(19);
		String content = stripper.getText(doc);
		return content;
	}

	

	public String readTextAsHtml(PDDocument doc) throws Exception {
		PDFText2HTML t2html = new PDFText2HTML();
		t2html.setStartPage(30);
		t2html.setEndPage(40);
		String content = t2html.getText(doc);
		return content;
	}

	/**
	 * This is a first pass filtering that can be applied universally
	 * 
	 * @param s
	 * @return
	 */
	public String generalReqFilter(String s) {
		// fix to not split sentences on abbreviations
		for (String abbrev : abbreviations) {
			String source = abbrev;
			String target = source.replace(".", " ");
			s = s.replace(source, target);
		}
		return s;
	}

	public String getHeaderNumber(String s) {
		return "";
	}

	/**
	 * Special filtering for the submarine pipelines systems document
	 * 
	 * @param s
	 * @return
	 */
	public List<String> filterSubmarinePipelineSystems(String s) {
		s = generalReqFilter(s);

		/*
		 * header numbering example: A 100 Objective
		 */
		String headerNumbering = "\\s([A-Z]\\s[0-9][0-9]?00.*[\n])"; // Ensure new line before and after a chapter
																		// header by adding a . and
		// splitting
		s = s.replaceAll(headerNumbering, "." + "$1" + ".");
		// System.out.println(s);

		String[] sents = s.split("\\.");
		int appendDigit = 1;
		String headerNumberWithAppendedDigit = null;
		// Requirements header: 203 This is a req ... ...
		Pattern reqHeader = Pattern.compile("^(\\d\\d\\d\\d?).*");
		String currentHeaderNumber = null;
		String sent2 = null;

		List<String> matches = new ArrayList<>();
		for (String sent : sents) {
//			System.out.println(sent);
			if (!sent.contains("Guidance note") && !sent.contains("---e-n-d")) {
				sent = sent.replaceAll("\\s+", " ").trim();
				Matcher reqHeaderMatcher = reqHeader.matcher(sent);
				if (reqHeaderMatcher.find()) {
//					System.out.println(sent);

					appendDigit = 1;
					currentHeaderNumber = reqHeaderMatcher.group(1);
					int end = reqHeaderMatcher.end(1);
//					System.out.println("number end: " + end + " length: " + sent.length());
					headerNumberWithAppendedDigit = currentHeaderNumber + "{0}|";

					// If the sent contains the next number - must break on that number - this
					// happens often (!)
					int nextHeaderNumber = Integer.parseInt(currentHeaderNumber) + 1;
					String nextHeaderNumberString = Integer.toString(nextHeaderNumber);
					Matcher nextHeaderMatcher = Pattern.compile(nextHeaderNumberString).matcher(sent);

					if (nextHeaderMatcher.find()) {
						String found = nextHeaderMatcher.group(0);
						int nextHeaderStart = nextHeaderMatcher.start(0);
						int nextHeaderEnd = nextHeaderMatcher.end(0);
						sent2 = found + "{0}|" + sent.substring(nextHeaderEnd, sent.length());
						sent2 = sent2.trim();
//						System.out.println(sent2);
						sent = sent.substring(0, nextHeaderStart);
//						System.out.println(sent);
//						System.out.println("Contains next header!!");					
					} else {
						sent2 = null;
					}

					sent = sent.substring(end, sent.length());
					sent = headerNumberWithAppendedDigit + sent.trim();
//					System.out.println(sent);
					// System.out.println(sent);

				} else if (currentHeaderNumber != null) {
					headerNumberWithAppendedDigit = currentHeaderNumber + "{" + appendDigit++ + "}|";
					sent = headerNumberWithAppendedDigit + sent.trim();
				} else {
					continue;
//					System.out.println(sent);
				}

				if (sent.contains(shallString)) {
//					System.out.println(sent);
					matches.add(sent);
				} else {
					// System.out.println(sent);
				}
				if (sent2 != null && sent2.contains(shallString)) {
					matches.add(sent2);
					// System.out.println("added sent2!!!");
					// System.out.println(sent2);
				}
				sent2 = null;

			} else {
				// System.out.println(m.group(0))
			}
		}

		return matches;

	}

	public List<String> filterRuShip(String s) {
		s = generalReqFilter(s);

		// Must first replace all dots within requirements numbering with space
		String reqNumberingWithDots = "\\s([0-9])\\.([0-9])\\s([0-9]).*";
		s = s.replaceAll(reqNumberingWithDots, "$1 $2 $3");
//		System.out.println(s);

		// Remove headerline
		String headerLine = "Rules for classification: Ships.*[\n]";
		s = s.replaceAll(headerLine, "\n");

		/*
		 * Chapter header numbering example: 1.1 Application
		 */
		String headerNumbering = "\\t([0-9]\\s[0-9]\\s[A-Z].*[\n])"; // Ensure new line before a chapter header
		s = s.replaceAll(headerNumbering, "." + "$1" + ".");
		// System.out.println(s);

		String[] sents = s.split("\\.");
		int appendDigit = 1;
		String headerNumberWithAppendedDigit = null;
		// Requirements header: 1.1.2 This is a req ... ...
		// But dots ate replaced with spaces in sentence
		Pattern reqHeader = Pattern.compile("^([0-9]\\s[0-9]\\s[0-9]).*");
		String currentHeaderNumber = null;

		List<String> matches = new ArrayList<>();
		for (String sent : sents) {
//			System.out.println(sent);
			if (!sent.contains("Guidance note:") && !sent.contains("---e-n-d")) {
				sent = sent.replaceAll("\\s+", " ").trim();

				// I should try to get the number, append a number to each sentence
				Matcher reqHeaderMatcher = reqHeader.matcher(sent);
				if (reqHeaderMatcher.find()) {
					appendDigit = 1;
					currentHeaderNumber = reqHeaderMatcher.group(1);
					int end = reqHeaderMatcher.end(1);
//					System.out.println("number end: " + end + " length: " + sent.length());
					headerNumberWithAppendedDigit = currentHeaderNumber + "{0}|";
					sent = sent.substring(end, sent.length());
					sent = headerNumberWithAppendedDigit + sent.trim();
//					System.out.println(sent);
					// System.out.println(sent);
				} else if (currentHeaderNumber != null) {
					headerNumberWithAppendedDigit = currentHeaderNumber + "{" + appendDigit++ + "}|";
					sent = headerNumberWithAppendedDigit + sent.trim();
					// System.out.println(sent);
				} else { /* This is the beginning of the document, just ignore */
					continue;
				}

				if (sent.contains(shallString)) {
//					System.out.println(sent);
					matches.add(sent);
				} else {
					// System.out.println(sent);
				}
			} else {
				// System.out.println(m.group(0))
			}
		}

		return matches;
	}



	

	public List<String> filterRequirementsHtml(String s) {
		String paragraphString = "<p>.*?</p>";
		String shallString = "shall";
		Pattern paragraphPattern = Pattern.compile(paragraphString, Pattern.DOTALL);

		Matcher m = paragraphPattern.matcher(s);
		List<String> matches = new ArrayList<>();
		while (m.find()) {
			String paragraph = m.group(0);
			if (paragraph.contains(shallString)) {
				System.out.println("**" + m.group(0) + "**");
				matches.add(m.group(0));
			} else {
				// System.out.println(m.group(0))
			}
		}

		return matches;
	}

	/**
	 * The procedure to read and extract thre requirements from DNV-OS-F101
	 * Sumbarine Pipeline Systems
	 * 
	 * @param document
	 * @return
	 */
	public List<String> readSubmarinePipelineSystems() throws Exception {
		int startPage = 0;
		int endPage = 370;
		String text = readTextByArea(reqDoc, startPage, endPage);
		reqs = filterSubmarinePipelineSystems(text);
		return reqs;
	}

	public List<String> readRuShip(int endPage) throws Exception {
		int startPage = 0;
		String text = readTextByArea(reqDoc, startPage, endPage);
		reqs = filterRuShip(text);
		return reqs;

	}

	public void printRequirements() {
		for (String req : reqs) {
			System.out.println(req);
		}
	}

	public void saveRequirements(String outPath) throws Exception {
		PrintWriter out = new PrintWriter(new File(outPath));
		for (String req : reqs) {
			out.println(req);
		}
		out.close();
		System.out.println("Requirements saved to " + outPath);

	}

	public static void main(String[] args) throws Exception {

		// String submarinePipelinePath = "/home/ole/Documents/DNVGL2.pdf";
		String ruShipPath = "/home/ole/Documents/dnvgl/ru-ship/dnvgl-st-f101.pdf";
		// String ruShipPath = "/home/ole/Documents/equinor/DocGo.Net-TR3032 Field
		// Instrumentation.pdf";
		String outPath = "/home/ole/Documents/dnvgl/dnvgl-ru-fd.txt";
		int lastPage = 100;

//		ReqExtractor submarinePipelineExtractor = new ReqExtractor(submarinePipelinePath);
//		submarinePipelineExtractor.readSubmarinePipelineSystems();
//		submarinePipelineExtractor.saveRequirements(outPath);
//		submarinePipelineExtractor.close();

		ReqExtractor ruShipExtractor = new ReqExtractor(ruShipPath);
		ruShipExtractor.readRuShip(lastPage);
		// ruShipExtractor.printRequirements();
		// ruShipExtractor.saveRequirements(outPath);
		ruShipExtractor.close();

	}
}
