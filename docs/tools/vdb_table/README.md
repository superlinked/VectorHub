# Vector DB Comparison

![](../../assets/tools/vdb_table/cover.gif)

[Vector DB Comparison](https://vdbs.superlinked.com/) is a free and open source tool from VectorHub to compare vector databases. It is created to outline the feature sets of different VDB solutions. Each of the features outlined has been verified to varying degrees.

This is a community initiative spearheaded by [Dhruv Anand](https://www.linkedin.com/in/dhruv-anand-ainorthstartech/), Founder of AI Northstar Tech, to give visibility to the different VDB offerings. Following the initial launch of the benchmarking table, a group of collaborators (listed below) was formed to verify claims before publishing on VectorHub.

[VectorHub](https://hub.superlinked.com/) is a community-driven learning platform for information retrieval hosted by [Superlinked](https://superlinked.com/). Superlinked is a vector compute solution in the ml stack alongside the different VDBs. 

For this exercise, the collaborators have worked with points of contact from the VBDs to ensure neutrality and fairness to create an accurate tool for practitioners.

**Table Interactions**
- Search: Use the search bar on top.
- Sort: Click on the column names to sort. Shift click multiple headers to sort by multiple columns in the order of clicks.
- Filter: Hover on column name to get the filter menu icon. Click the relevant value to filter.
- Vendor links: Each vendor has links to their website, github, documentation, discussion (on this github repo) and point of contact. Click the link button next to the vendor name.
- Documentation links: Cells which have supporting link to vendor's documentation have an external link button in the cell.
- Comments: Additional comments by maintainers, are shown on hovering over a cell. 


**Maintainers:**
- [Dhruv Anand](https://www.linkedin.com/in/dhruv-anand-ainorthstartech/)
- [Prashanth Rao](https://www.linkedin.com/in/prrao87/)
- [Ravindra Harige](https://www.linkedin.com/in/ravindraharige/)
- [Daniel Svonava](https://www.linkedin.com/in/svonava/)

**Frontend:**
- [Arunesh Singh](https://www.linkedin.com/in/aruneshsingh99/)



## Contributing

Thanks for your interest in contributing to [vdbs.superlinked.com](https://vdbs.superlinked.com) and keeping the data upto date. 

We use [discussions](https://github.com/superlinked/VectorHub/discussions/categories/vdb-comparison) as our way to have conversations about each vendor. Please find the relevant discussion and add to the conversation.

Kindly review the following sections before you submit your issue or initial pull request, and use the approriate issues/PR template. In addition, check for existing open issues and pull requests to ensure that someone else has not already corrected the information.

If you need any help, feel free to tag [@AruneshSingh](https://github.com/AruneshSingh) in your discussions/issues/PRs.

### About this repository

- **Frontend:** The frontend is created in React, using ag-grid for the tables and Material UI for the interface components. It's hosted and deployed using Vercel.
- B**ackend:** This github serves as the data for the table. Any updates to the JSON files are validated using github actions and pushed to a Google storage bucket, from where the frontend fetches the data.
- **Discussions:** All the vendors have a dedicated discussion [here](https://github.com/superlinked/VectorHub/discussions/categories/vdb-comparison). Please ensure to go through the discussions before raising an issue/PR.

### Structure

This subdirectory is structured as follows:

```
tools/
└── vdb_table/
    ├── vendor.schema.json
    ├── ...
    ├── ...
    └── data/
        ├── vendor1.json
        ├── vendor2.json
        ├── vendor3.json
        └── ...
```

| File                  | Description                              |
| --------------------- | ---------------------------------------- |
| `vendor.schema.json`  | JSON Schema file that describes the attributes, it's properties, and description |
| `vendorX.json`        | All the attribute data for vendor X. We have one file per vendor.    |


Attributes inside vendorX.json has the following properties
- `support`: Whose values can be `[ "", "none", "partial", "full" ]` indicating on confidence levels, for that attribute support. NOTE: Each change where a "support" claim is being added MUST include  either i) a reference to documentation, ii) an example of the functionality being described, or iii) a link to the actual code implementing the specific functionality.
    - `""` means the cell will be blank. 
    - `"none"` means the cell will have a ❌. 
    - `"partial"` means the cell will have a 🟨.
    - `"full"` means the cell will have a ✅.
- `value`: `license` and `dev_languages` have this property to support values about license details and languages (as a list).
- `source_url`: To provide documentation links, or evidence supporting the attribute values. It is shown as the 'external link' button in the cell.
- `comment`: Any other useful information that will be shown on hover and with the info icon.
- `unlimited`: `doc_size` and `vector_dims` can have this property set true if they support unlimited values.

For more nuanced information about each property, have a look at `vendor.schema.json`.




