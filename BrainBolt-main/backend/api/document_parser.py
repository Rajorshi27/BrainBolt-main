
class DocumentsProcessor:
    """

    This class provides methods for loading and deleting documents,
    creating and storing embeddings in a vector database,
    and checking if files uploaded have changed
    """

    def __init__(self, directory, vector_db) -> None:
        self.directory = directory
        self.vector_db = vector_db

    @property
    def directory(self):
        """Return the directory for processing documents"""
        return self._directory

    @directory.setter
    def directory(self, value):
        """Set the directory for processing documents"""
        if value is not None:
            if isinstance(value, str):
                if value == "":
                    raise ValueError(
                        "Directory for DocumentProcessor cannot be an empty string"
                    )
                else:
                    self._directory = value
                    # Create the directory only if directory does not exist
                    if os.path.isdir(value):
                        print("The directory has already been created")
                    else:
                        os.mkdir(value)
            else:
                raise TypeError(
                    "Directory value for DocumentProcessor must be a str")
        else:
            raise TypeError("Directory for DocumentProcessor is null")

    @property
    def vector_db(self):
        """Returns the vector_db directory name for storing vectors"""
        return self._vector_db

    @vector_db.setter
    def vector_db(self, value):
        """Set the vector_db directory name"""
        if value is not None:
            if isinstance(value, str):
                if value == "":
                    raise ValueError(
                        "Vector Directory for DocumentProcessor cannot be an empty string"
                    )
                else:
                    self._vector_db = value
                    # Create the directory only if directory does not exist
                    if os.path.isdir(value):
                        print("The vector directory has already been created")
                    else:
                        os.mkdir(value)
            else:
                raise TypeError(
                    "Vector Directory value for DocumentProcessor must be a str")
        else:
            raise TypeError("Vector Directory for DocumentProcessor is null")

    def generate_file_hash(self, file_path):
        """Generates the hash content of a file
        Args:
            file_path (str): Contains the file path
        Returns:
            hasher.hexdigest (str): The final hash value in a string format
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as file:
            # This reads the file in chunks (8192 bytes at a time)
            while chunk := file.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_hashes_for_uploaded_files(self):
        """Reads the files in directory and generates an array of hashes.
        Generating hashes indicate whether new files have been added or if files have been changed
        Args:
            None
        Returns:
            file_contents (dict): A dictionary consisting of hash values
        """
        file_contents = {}
        # Check if files exist inside directory
        if len(os.listdir(self.directory)) != 0:
            files = os.listdir(self.directory)
            for each_file in files:
                print(f"Attempting to generate hashes for {each_file}:")
                try:
                    file_path = os.path.join(self.directory, each_file)
                    # Check if the file_path is a file
                    if os.path.isfile(file_path):
                        hash_content = self.generate_file_hash(file_path)
                        file_contents[each_file] = hash_content
                        print(
                            f"***Hashes were successfully generated for {each_file}")
                    else:
                        raise IsADirectoryError(
                            "Error! The path given is a directory, not a file!")
                except FileNotFoundError:
                    print("File does not exist")
        else:
            print("The directory is empty! Please upload some files")
        return file_contents

    def save_uploaded_file(self, uploaded_file):
        # Check if the 'data' folder exists, and create it if not
        data_folder = self.directory
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as open_file:
            open_file.write(uploaded_file.getbuffer())
        return st.success(f"Saved file {uploaded_file.name} to {data_folder}.")

    def generate_embeddings(_self, _db, _embeddings):  # pylint: disable=no-self-argument
        """Generates embeddings for documents inside directory
        Args: 
            None
        Returns: 
            embeddings: contains embeddings 
        """
        # persist_directory is the local vector directory where embeddings will be saved
        persist_directory = _self.vector_db

        # Add embeddings
        if len(_self.list_of_uploaded_files()) != 0:
            files_in_chroma_vector_db = _self.list_of_files_in_chroma_vector_db(
                _db)
            print(f"The files inside CHROMA DB: {files_in_chroma_vector_db}")
            # If Chroma vector db is not empty
            if files_in_chroma_vector_db:
                # Get files that have not been processed yet
                list_of_files_not_in_chroma_db = _self.get_uploaded_files_not_in_chroma_db(
                    _db)

                # Multi processing
                with ProcessPoolExecutor() as executor:
                    for each_file in list_of_files_not_in_chroma_db:
                        print(
                            f"Generating embeddings for file not in chroma db {each_file}")
                        # Get file_path
                        file_path = os.path.join(_self.directory, each_file)
                        # Check if file exist
                        if os.path.isfile(file_path):
                            # Use UnstructuredFileLoader
                            loader = UnstructuredFileLoader(file_path)

                            # Load the documents
                            documents = loader.load()

                            # Split the documents
                            text_splitter = CharacterTextSplitter(
                                chunk_size=2500, chunk_overlap=800)

                            # Create chunks
                            chunks = text_splitter.split_documents(documents)

                            # Add the chunks to the Chroma DB vector database
                            _db = Chroma.from_documents(
                                documents=chunks, embedding=_embeddings, persist_directory=persist_directory)

                            # Save the changes to disk
                            _db.persist()
                            print(
                                f"**********Successfully generated embedding for {each_file}**********")
                retriever = _db.as_retriever(
                    search_type="similarity", search_kwargs={"k": 3})
                return retriever

            # This block is called when Chroma has no files at all
            else:
                print(f"ChromaDB vector database is empty! All files will be loaded!")
                # Create loader
                loader = DirectoryLoader(
                    _self.directory, use_multithreading=True)
                # Load all documents
                documents = loader.load()
                # Create TextSplitter
                text_splitter = CharacterTextSplitter(
                    chunk_size=2500, chunk_overlap=800)
                # Generate the chunks
                chunks = text_splitter.split_documents(documents)
                # Set DB
                _db = Chroma.from_documents(
                    chunks, _embeddings, persist_directory=persist_directory
                )
                # Save the db contents
                _db.persist()
                # Retrieve the data
                retriever = _db.as_retriever(
                    search_type="similarity", search_kwargs={"k": 3})
                return retriever

    def force_generate_embeddings(self):
        """This function will force generate embeddings!"""
        # persist_directory is the local vector directory where embeddings will be saved
        persist_directory = self.vector_db
        embeddings = HuggingFaceEmbeddings()

        # Create loader
        loader = DirectoryLoader(self.directory, use_multithreading=True)
        # Load all documents
        documents = loader.load()
        # Create TextSplitter
        text_splitter = CharacterTextSplitter(
            chunk_size=2500, chunk_overlap=800)
        # Generate the chunks
        chunks = text_splitter.split_documents(documents)
        # Set DB
        _db = Chroma.from_documents(
            chunks, embeddings, persist_directory=persist_directory
        )
        # Save the db contents
        _db.persist()
        # Retrieve the data
        retriever = _db.as_retriever(
            search_type="similarity", search_kwargs={"k": 3})
        return retriever

    def delete_entire_directory(self):
        """Deletes the directory and all its files/folders"""
        # Check if dir exists
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
        else:
            if os.path.isfile(self.directory):
                raise FileExistsError("Error! Directory path is a file!")

    def delete_directory_contents(self):
        """Deletes the contents of the directory only"""
        # Check if the path exists
        if os.path.isdir(self.directory):
            # Obtain list of files and folders inside directory
            list_of_files_and_folders = os.listdir(self.directory)
            for content in list_of_files_and_folders:
                content_path = os.path.join(self.directory, content)
                # Check if its a file
                if os.path.isfile(content_path):
                    # Remove file
                    try:
                        os.remove(content_path)
                    except FileNotFoundError:
                        print("Error! Could not delete file!")
                elif os.path.isdir(content_path):
                    try:
                        shutil.rmtree(content_path)
                    except NotADirectoryError:
                        print("Error! Could not delete directory!")
                else:
                    raise OSError("Error! Could not delete file or path")
        else:
            print("Directory does not exist! Please upload a file")

    # This does not work as the vector_db folder has a process inside that has to be killed first!
    def delete_vector_db(self):
        """This method removes the vector database path"""
        # Check if vector database exist
        if os.path.isdir(self.vector_db):
            shutil.rmtree(self.vector_db)
        else:
            if os.path.isfile(self.vector_db):
                raise FileExistsError("Error! Directory path is a file!")

    def list_of_uploaded_files(self):
        """This method simply returns an array of currently uploaded files"""
        # Check if our directory where files are uploaded exists
        files = []
        if os.path.isdir(self.directory):
            list_of_files_in_directory = os.listdir(self.directory)
            for each_file in list_of_files_in_directory:
                # Obtain path
                file_path = os.path.join(self.directory, each_file)
                # Check if file is a file
                if os.path.isfile(file_path):
                    # Append file to files array
                    files.append(each_file)
        else:
            raise NotADirectoryError(
                "Error! Could not find files, directory does not exist!")
        return files

    def list_of_files_in_chroma_vector_db(self, db):
        """Returns a list of files in Chroma vector db
        Args:
            db: The Chroma DB Vector database
        Returns:
            files (list): A list of files in the Chroma vector database
        """
        # Store files in list
        files = []
        # Get all files in Chroma vector db
        docs = db.get()

        # Check if Chroma vector db is empty!
        try:
            sources = [metadata['source'] for metadata in docs['metadatas']]

            # Get rid of duplicates
            files = list(dict.fromkeys(sources))

            # Remove 'uploaded_documents\\' from each file path
            files_only = [file.replace('uploaded_documents\\', '')
                          for file in files]

            return files_only
        except KeyError:
            print("Critical Error! Could not find list of files in chroma vector db!")
        return files

    def get_uploaded_files_not_in_chroma_db(self, db):
        """Returns a list of files that are in the uploaded directory but not in the Chroma Vector Database.
        Args:
            db: The Chroma DB Vector database
        Returns:
            list(list_of_files_not_in_chroma_db): A list of files not in the chroma db vector database but are in the uploaded directory
        """
        # List of files in Chroma Vector Database
        list_of_files_in_chroma_vector_db = self.list_of_files_in_chroma_vector_db(
            db)
        # List of uploaded files
        list_of_uploaded_files = self.list_of_uploaded_files()

        set_of_files_in_chroma_vector_db = set(
            list_of_files_in_chroma_vector_db)
        set_of_uploaded_files = set(list_of_uploaded_files)

        list_of_files_not_in_chroma_db = set_of_uploaded_files - \
            set_of_files_in_chroma_vector_db
        return list(list_of_files_not_in_chroma_db)

    def download_file_from_directory(self, file_name):
        """Downloads a file from the local directory"""
        return None

    def get_file_text(self, file_name, db):
        """Returns all the text from a file"""

        # Validation
        if isinstance(file_name, str):
            # See if file exists

            # Get file path
            file_path = os.path.join(self.directory, file_name)

            results = db.get(where={"source": file_path})
            all_text = ""
            if 'documents' in results:
                if len(results['documents']) > 0:
                    for each_document in results['documents']:
                        all_text += each_document

            return all_text
