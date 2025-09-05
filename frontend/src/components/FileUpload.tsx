import React, { useCallback } from 'react';

export interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedFormats: string[];
  maxSize: number; // MB
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, acceptedFormats, maxSize }) => {
  const onChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    if (!acceptedFormats.includes(ext)) {
      alert(`Invalid file type. Allowed: ${acceptedFormats.join(', ')}`);
      return;
    }
    if (f.size > maxSize * 1024 * 1024) {
      alert(`File too large. Max ${maxSize} MB`);
      return;
    }
    onFileSelect(f);
  }, [onFileSelect, acceptedFormats, maxSize]);

  return (
    <input
      type="file"
      onChange={onChange}
      className="input cursor-pointer file:mr-4 file:rounded-md file:border-0 file:bg-blue-600 file:px-3 file:py-2 file:text-white hover:file:bg-blue-700"
    />
  );
};

export default FileUpload;
