package req_extract;

import java.awt.geom.Rectangle2D;
import java.io.File;
import java.io.PrintWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.text.PDFTextStripperByArea;

public class PDFParser {

	private String[] abbreviations = new String[] { "Sec.", "Sect.", "i.e.", "e.g.", "I.e.", "E.g.", "Etc.", "etc.", "Techn.", "Ch.", "ch.", "Pt."};

	public String pdfPath;
	public File pdfFile;
	public String xmlPath;
	public PDDocument pdfDocument;



	public PDFParser(String PDFPath, String xmlPath) throws Exception {
		this.pdfPath = PDFPath;
		this.pdfFile = new File(PDFPath);
		this.pdfDocument = PDDocument.load(pdfFile);
		this.xmlPath = xmlPath;
	}

	public void writeXML(String s) {
		try (PrintWriter fileWriter = new PrintWriter(xmlPath)) {
			fileWriter.write(s);
		} catch (Exception e) {
			System.out.println("Cannot write to /home/ole/Docuemnts/req.xml");
			System.exit(-1);
		}
		System.out.println("Successfully printed xml to " + xmlPath);
	}


	/**
	 * Reading only the inner part of the document excluding header and footer
	 *
	 */
	public String readTextByArea(PDDocument doc, int startPage, int endPage) throws Exception {

		// Equinor has top text 0-70
		// content in 70-450
//		int headerHeight = 70;
//		int pageHeight = 450;

		// DNVGL-ST-F101 has top text 0-50
		// content from 50-700
		int headerHeight = 50;
		int pageHeight= 650;
		int headerStart = 0;
		int width = 550;
		//int height = 50;


		int footerStart = headerHeight + pageHeight;

		Rectangle2D headerRegion = new Rectangle2D.Double(0, headerStart, width, headerHeight);
		Rectangle2D footerRegion = new Rectangle2D.Double(0, footerStart, width, headerHeight);
		Rectangle2D contentRegion = new Rectangle2D.Double(0, headerStart + headerHeight, width, pageHeight);

		String headerRegionName = "header";
		String footerRegionName = "footer";
		String contentRegionName = "content";

		PDFTextStripperByArea stripper = new PDFTextStripperByArea();
		stripper.setParagraphStart("\t");
		stripper.setSortByPosition(true);
		stripper.addRegion(headerRegionName, headerRegion);
		stripper.addRegion(footerRegionName, footerRegion);
		stripper.addRegion(contentRegionName, contentRegion);

		String content = "";
		for (int pageNumber = startPage; pageNumber < endPage; pageNumber++) {
			PDPage page = doc.getPage(pageNumber + 1);
			stripper.extractRegions(page);
			content += stripper.getTextForRegion(contentRegionName);
		}
		return content;
	}

	/**
	 * This removes . from common abbreviations
	 *
	 * @param s
	 * @return
	 */
	public String abbreviationFilter(String s) {
		// fix to not split sentences on abbreviations
		for (String abbrev : abbreviations) {
			String source = abbrev;
			String target = source.replace(".", " ");
			s = s.replace(source, target);
		}
		return s;
	}


	public String parseRequirement(String s) {
		boolean sub2 = false; /* Set true if requirements are on the form 2.2.2.2, false if they are on 2.2.2 */
		boolean usePart = true;

		s = abbreviationFilter(s);
		System.out.println("Replacing non-xml characters");
		// Remove <, > and & symbols (to parse xml)
		s = s.replace(">", "&gt;");
		s = s.replace("<", "&lt;");
		s = s.replace("&", "and");


		System.out.println("Identifying sections");
		/* SECTION 4 DESIGN - LOADS */
		/* identify section headers */
		String sectionHeader = "SECTION\\s(\\d\\d?)\\s(.*)";
		s = s.replaceAll(sectionHeader, "<section num=\"$1\" title=\"$2\">");

		System.out.println("Identifying appendix sections");
		/* APPENDIX A TYPES AND MINIMUM DIMENSIONS */
		String appendixSectionHeader = "APPENDIX\\s(\\w)\\s(.*)";
		s = s.replaceAll(appendixSectionHeader, "<section num=\"$1\" title=\"$2\">");


		/* 1 Material requirements */
		/* identify part haders: This is a level below section but before subsections.
		 * To avoid enumerate environment we have to check it doesn't end with a DOT */
		if (usePart) {
			System.out.println("Identifying parts");
			String partHeader = "^.(\\d\\d?)\\s+([A-Z].*)[^.]$"; /* note: added + */
			Matcher partHeaderMatcher = Pattern.compile(partHeader, Pattern.MULTILINE).matcher(s);
			s = partHeaderMatcher.replaceAll("\n<part num=\"$1\" title=\"$2\">");
		}


		System.out.println("Identifying sub1");
		/* sub1: 4 1 General */
		String subsec1Header = "\\t(\\d\\d?\\.\\d\\d?)\\s+([A-Z].*)"; /*note: added + */
		//s = s.replaceAll(subsec1Header, "<sub1 num=$1>$2</sub1>");
		s = s.replaceAll(subsec1Header, "<sub1 num=\"$1\" title=\"$2\">");

		/* sub1: A 1 General */
		String appendix1Header = "\\t(\\w\\.\\d\\d?)\\s+([A-Z].*)";
		s = s.replaceAll(appendix1Header, "<sub1 num=\"$1\" title=\"$2\">");


		/* We should identify these first */
		System.out.println("Identifying req (with 4 digits)");
		/* 4 1 1 1 Requirement */
		String req_num4 = "(\\d\\d?\\.\\d\\d?\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z/\n].*)"; /* note: added+ */
		s = s.replaceAll(req_num4, "<req num=\"$1\">$2");

		if (sub2) {
			System.out.println("Identifying sub2");
			/* sub2: 4 1 1 Objective */
			//String subsec2Header = "\\t(\\d\\d?\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z].*)";
			//s = s.replaceAll(subsec2Header, "<sub2 num=\"$1\" title=\"$2\">");

			String req_num = "^\\t?(\\d\\d?\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z/].*)"; /* note: added start of line anchor */
			Pattern req_num_pattern= Pattern.compile(req_num, Pattern.MULTILINE);
			Matcher req_num_matcher = req_num_pattern.matcher(s);
			s = req_num_matcher.replaceAll("<sub2 num=\"$1\" title=\"$2\">$2");



			/* sub2: A 1 1 Objective */
			String appendix2Header= "\\t(\\w\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z].*)";
			s = s.replaceAll(appendix2Header, "<sub2 num=\"$1\" title=\"$2\">");
		} else {
			System.out.println("Identifying req");
			/* 4 1 1 Requirement */
//			String req_num = "(\\d\\d?\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z/].*)"; /* note: added + for RU-ship */
//			s = s.replaceAll(req_num, "<req num=\"$1\">$2");

			String req_num = "^(\\d\\d?\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z/].*)"; /* RU-shipCh4Pt8, pattern need start of line anchor */
			Pattern req_num_pattern= Pattern.compile(req_num, Pattern.MULTILINE);
			Matcher req_num_matcher = req_num_pattern.matcher(s);
			s = req_num_matcher.replaceAll("<req num=\"$1\">$2");

			/* A 1 1 Requirement */
			String appendix_req_num = "(\\w\\.\\d\\d?\\.\\d\\d?)\\s+([A-Z/].*)"; /*note: added + for RU-SHIP */
			s = s.replaceAll(appendix_req_num, "<req num=\"$1\">$2");
		}



		/* A 1 1 1 Requirement */
		String appendix_req_num = "(\\w\\.\\d\\d?\\.\\d\\d?\\.\\d\\d?)\\s([A-Z].*)";
		s = s.replaceAll(appendix_req_num, "<req num=\"$1\">$2");

		System.out.println("Removing tab characters");
		/* remove all tab characters */
		s = s.replaceAll("\t", "");

		System.out.println("Identifying tables");
		/* Table 1-1 */
		String tab_num_double = "\\nTable\\s(\\d\\d?-\\d\\d?)(.*)";
		s = s.replaceAll(tab_num_double, "\n<table num=\"$1\">$2");

		/* Table A-1 */
		String appendix_tab_num_double = "^Table\\s(\\w-\\d\\d?)\\s([A-Z].*)";
		Pattern appendix_tab_pat = Pattern.compile(appendix_tab_num_double, Pattern.MULTILINE);
		Matcher appendix_tab_matcher = appendix_tab_pat.matcher(s);
		s = appendix_tab_matcher.replaceAll("\n<table num=\"$1\">$2");

		/* Table 1 */
		String tab_num_single = "\\nTable\\s(\\d\\d?)(.*)";
		s = s.replaceAll(tab_num_single, "\n<table num=\"$1\">$2");

		System.out.println("Identifying figures");
		/* Figure 1-1 */
		String fig_num_double = "\\nFigure (\\d-\\d)\\s([A-Z].*)";
		s = s.replaceAll(fig_num_double, "\n<figure num=\"$1\">$2</figure>");

		/* Figure 1 */
		String fig_num_single = "\\nFigure (\\d\\d?)\\s([A-Z].*)";
		s = s.replaceAll(fig_num_single, "\n<figure num=\"$1\">$2</figure>");


		System.out.println("Identifying guidance notes");
		/* Guidance notes */
		s = s.replaceAll("Guidance note:", "<guidancenote>");
		s = s.replaceAll("Guidance note\\s(\\d)", "<guidancenote>");
		String guidanceNoteEnd = "---e-n-d---o-f---g-u-i-d-a-n-c-e---n-o-t-e---";
		String guidanceNoteEnd2 = "---e-n-d---of---g-u-i-d-a-n-c-e---n-o-t-e---";
		s = s.replaceAll(guidanceNoteEnd, "</guidancenote>");
		s = s.replaceAll(guidanceNoteEnd2, "</guidancenote>"); // note: added for RU-ship pt2ch3
		s = s.replaceAll("<guidancenote>\\n", "<guidancenote>");
		s = s.replaceAll("\\n\\s</guidancenote>", "</guidancenote>");



		System.out.println("Remvoing everything before section 1");
		/* remove everything before section 1 */
		String sec1 = ".*?(<section num=\"1\".*)";
		Pattern sec1_pat = Pattern.compile(sec1, Pattern.MULTILINE | Pattern.DOTALL);
		Matcher sec1_matcher = sec1_pat.matcher(s);
		s = sec1_matcher.replaceFirst("$1"); // This does not work with Equinor-documents

		System.out.println("Adding mandatory xml-elements");
		/* adding xml elements */
		s = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n<document>\n" + s;
		s = s + "</document>";


		System.out.println("Identifying end of table");
		/* END of table */
		String table_end = "(<table.*?>.*?)(?=(<document|<part|<section|<sub|<req|<figure|<table|<guidancenote|</guidancenote))";
		Pattern table_end_pat = Pattern.compile(table_end, Pattern.DOTALL);
		Matcher table_end_matcher = table_end_pat.matcher(s);
		s = table_end_matcher.replaceAll("$1</table>\n");
		s = s.replaceAll("\\n</table>", "</table>");

		System.out.println("Identifying end of req");
		/* END OF req*/
		String req_end = "(<req.*?>.*?)(?=(<section|<part|<sub|<req|</document))";
		Pattern req_end_pat = Pattern.compile(req_end, Pattern.DOTALL);
		Matcher req_end_matcher = req_end_pat.matcher(s);
		s = req_end_matcher.replaceAll("$1</req>\n");

		System.out.println("Remove newline before </req>");
		/* remove <newline> before </req> */
		s = s.replaceAll("\\n</req>", "</req>");


		System.out.println("Identifying end of table if it is within a requirement"); /* note: added this for RU-ship, doesn't always work*/
		/* END of table */
		String req_table_end = "(<req.*?>.*<table.*?>.*?)(?=</req)";
		Pattern req_table_end_pat = Pattern.compile(req_table_end, Pattern.DOTALL);
		Matcher req_table_end_matcher = req_table_end_pat.matcher(s);
		//s = req_table_end_matcher.replaceAll("$1</table>\n");
		//s = s.replaceAll("\\n</table>", "</table>");

		if (sub2) {
			System.out.println("Identifying end of sub2");
			/* END OF sub2 */
			String sub2End = "(<sub2.*?>)(.*?)(?=(<sub1|<part|<sub2|<section|</document))"; // end with any of [sub1, sub2, section]
			Matcher sub2_end_matcher = Pattern.compile(sub2End, Pattern.DOTALL).matcher(s);
			s = sub2_end_matcher.replaceAll("$1$2</sub2>\n");
		}

		System.out.println("Identifying end of sub1");
		/* END OF sub1 */
		String sub1End = "(<sub1.*?>)(.*?)(?=(<sub1|<part|<section|</document))"; // end with [sub1 or section]
		Matcher sub1_end_matcher = Pattern.compile(sub1End, Pattern.DOTALL).matcher(s);
		s = sub1_end_matcher.replaceAll("$1$2</sub1>\n");

		System.out.println("Identifying end of part");
		/* END OF part */
		String partEnd = "(<part.*?>)(.*?)(?=(<part|<section|</document))"; // end with [part or section]
		Matcher part_end_matcher = Pattern.compile(partEnd, Pattern.DOTALL).matcher(s);
		s = part_end_matcher.replaceAll("$1$2</part>\n");

		System.out.println("Identifying end of section");
		/* identify end of section */
		String sectionEnd = "(<section.*?>)(.*?)(?=(<section|</document))";
		Matcher section_end_matcher = Pattern.compile(sectionEnd, Pattern.DOTALL).matcher(s);
		s = section_end_matcher.replaceAll("$1$2</section>\n");

		System.out.println("Parsing PDF to XML finished... ...");
		return s;
	}

	public String parseDocument(int endPage) throws Exception {
		int startPage = 0;
		String text = readTextByArea(pdfDocument, startPage, endPage);
		return parseRequirement(text);
	}

	public static void main(String[] args) throws Exception {
		/*
		 * Remember: Have to manually define:
		 * 	* If the document has sub2-level (req on 2.2.2.2) or sub1-level (req on 2.2.2)
		 */

		String pdfPath = "../../data/DNVGL-ST-F101.pdf";
		//String pdfPath = "/path/to/DNVGL-RU-SHIP.pdf";

		String outPath = "../../xml/DNVGL-ST-F101.xml";

		int lastPage = 319;
		PDFParser parser = new PDFParser(pdfPath, outPath);
		String xmlString = parser.parseDocument(lastPage);
		parser.writeXML(xmlString);

	}
}
